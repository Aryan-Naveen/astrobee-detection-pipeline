#!/usr/bin/env python3

from __future__ import division


# ROS imports
import rospy
import std_msgs.msg
from rospkg import RosPack
from std_msgs.msg import UInt8
from sensor_msgs.msg import Image as ROSImage
from msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Polygon, Point32


# Python imports
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import os, cv2

from utils.undistorter import Undistorter

# Deep learning imports
import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision import transforms
convert_tensor = transforms.ToTensor()

# Get package path in file directory
package = RosPack()
package_path = package.get_path('handrail_segmentation')


def get_trained_model(weights_path, num_classes = 5):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    # replace the pre-trained head with a new one
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)


    model.load_state_dict(torch.load(weights_path))

    return model

def filter_preds(bboxs, masks, labels, scores, thresh=0.7):
    f_bboxs = []
    f_masks = []
    f_labels = []
    for bbox, mask, label, score in zip(bboxs, masks, labels, scores):
        if score > thresh:
            f_bboxs.append(bbox.reshape(4,))
            f_masks.append(mask.reshape(240, 320))
            f_labels.append(label)

    return f_bboxs, f_masks, f_labels


class HandrailDetectorManager():
    def __init__(self):
        # Load weights parameter
        weights_name = rospy.get_param('~weights_name', 'yolov3_ckpt_100.pth')
        self.weights_path = os.path.join(package_path, 'src/models/weights', weights_name)
        rospy.loginfo("Found weights, loading %s", self.weights_path)

        # Raise error if it cannot find the model
        if not os.path.isfile(self.weights_path):
            raise IOError(('{:s} not found.').format(self.weights_path))

        # Load image parameter and confidence threshold
        self.image_topic = rospy.get_param('~image_topic', '/sample_img')
        self.nms_th = rospy.get_param('~nms_th', 0.5)
        #
        # # Load publisher topics
        self.segmentation_mask = rospy.get_param('~detected_objects_topic', "det/features")
        self.published_annotated_img_topic = rospy.get_param('~detections_image_topic', "det/features/annotated_img")
        self.publish_image = rospy.get_param('~publish_image', True)

        # Initialize width and height
        self.h = 0
        self.w = 0

        self.model = get_trained_model(self.weights_path)
        self.model.eval() # Set in evaluation mode
        rospy.loginfo("Deep neural network loaded")

        dims = np.array([320, 240])
        K = np.array([
          [608.8073, 0.0, 632.53684],
          [0.0, 607.61439, 549.08386],
          [0.0, 0.0, 1.0]])
        dist = 0.998693

        self.undist_ = Undistorter(dims, K, dist)

        # Define subscribers
        self.image_sub = rospy.Subscriber(self.image_topic, ROSImage, self.imageCb, queue_size = 1, buff_size = 2**24)

        # Define publishers
        self.pub_ = rospy.Publisher(self.segmentation_mask, ROSImage, queue_size=10)
        self.pub_viz_ = rospy.Publisher(self.published_annotated_img_topic, ROSImage, queue_size=10)
        rospy.loginfo("Launched node for handrail segmentation")

    def imageCb(self, data):
        # Convert the image to OpenCV
        self.cv_image = self.undist_.undistort(np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)[:,:,::-1])
        self.h = data.height
        self.w = data.width

        # Configure input
        input_img = self.preprocessImage(self.cv_image)

        # set image type
        if torch.cuda.is_available():
            input_img = input_img.to("cuda")

        # Get detections from network
        detections = self.model(input_img)

        # Parse detections
        if detections[0] is not None:
            for detection in detections:
                bboxs = detection['boxes'].detach().numpy()
                masks = detection['masks'].detach().numpy()
                labels = detection['labels'].detach().numpy()
                scores = detection['scores'].detach().numpy()

                bboxs, masks, labels = filter_preds(bboxs, masks, labels, scores)
                rospy.loginfo(f"\t+ Label: {self.classes[int(det_class)]} | Confidence: {conf:0.4f} | Center: ({det_msg.bbox.center.x}, {det_msg.bbox.center.y}) | Size: ({det_msg.bbox.size_x}, {det_msg.bbox.size_y})")


            # Publish detection results
            self.pub_.publish(detections_msg)

            # Visualize detection results
            if (self.publish_image):
                self.visualizeAndPublish(detections_msg, self.cv_image)
        else:
            rospy.loginfo("No detections available, next image")
        return True


    def preprocessImage(self, imgIn):
        return [convert_tensor(imgIn)]

    def visualizeAndPublish(self, output, imgIn):
        # Copy image and visualize
        imgOut = np.ascontiguousarray(imgIn)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        thickness = int(2)
        for index in range(len(output.detections)):
            label = output.detections[index].results[0].id
            x_p1 = output.detections[index].bbox.center.x - 0.5*output.detections[index].bbox.size_x
            y_p1 = output.detections[index].bbox.center.y - 0.5*output.detections[index].bbox.size_y
            x_p3 = output.detections[index].bbox.center.x + 0.5*output.detections[index].bbox.size_x
            y_p3 = output.detections[index].bbox.center.y + 0.5*output.detections[index].bbox.size_y
            confidence = output.detections[index].results[0].score

            # Find class color
            if label in self.classes_colors.keys():
                color = self.classes_colors[label]
            else:
                # Generate a new color if first time seen this label
                color = np.random.randint(0,255,3)
                self.classes_colors[label] = color

            # Create rectangle
            start_point = (int(x_p1), int(y_p1))
            end_point = (int(x_p3), int(y_p3))
            lineColor = (int(color[0]), int(color[1]), int(color[2]))

            cv2.rectangle(imgOut, start_point, end_point, lineColor, thickness)
            text = ('{:s}: {:.3f}').format(self.classes[int(label)],confidence)
            cv2.putText(imgOut, text, (int(x_p1), int(y_p1+20)), font, fontScale, (180,0,180), thickness ,cv2.LINE_AA)

        # Publish visualization image
        print(len(output.detections))
        cv2.imwrite('/home/aryan/Documents/nasa_ws/astrobee-detection-pipeline/output.png', imgOut)
        new_img_msg = ROSImage()
        new_img_msg.header = output.header
        new_img_msg.height = self.h
        new_img_msg.width = self.w
        new_img_msg.encoding = 'bgr8'
        new_img_msg.step = new_img_msg.width*3
        new_img_msg.data = np.array(imgOut)[:,:,::-1].tobytes()
        self.pub_viz_.publish(new_img_msg)


if __name__=="__main__":
    # Initialize node
    rospy.init_node("handrail_detector_node")

    # Define detector object
    dm = DetectorManager()

    # Spin
    rospy.spin()
