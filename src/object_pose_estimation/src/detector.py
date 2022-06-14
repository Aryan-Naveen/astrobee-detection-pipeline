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
import numpy as np
import scipy.io as sio
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os, cv2, time
from skimage.transform import resize
from torch.cuda import is_available
from utils.undistorter import Undistorter

package = RosPack()
package_path = package.get_path('object_pose_estimation')

# Deep learning imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models.models import Darknet
from utils.utils import *
from utils.transforms import Resize, DEFAULT_TRANSFORMS

import torchvision.transforms as transforms
from pytorchyolo.models import load_model

# Detector manager class for YOLO
class DetectorManager():
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
        self.confidence_th = rospy.get_param('~confidence', 0.5)
        self.nms_th = rospy.get_param('~nms_th', 0.5)
        #
        # # Load publisher topics
        self.detected_objects_topic = rospy.get_param('~detected_objects_topic', "det/features")
        self.published_image_topic = rospy.get_param('~detections_image_topic', "det/features/annotated_img")

        # Load other parameters
        config_name = rospy.get_param('~config_name', 'yolov3-custom.cfg')
        self.config_path = os.path.join(package_path, 'src/models/config', config_name)
        classes_name = rospy.get_param('~classes_name', 'obj.names')
        self.classes_path = os.path.join(package_path, 'src/models/config', classes_name)
        self.gpu_id = rospy.get_param('~gpu_id', 0)
        self.network_img_size = rospy.get_param('~img_size', 416)
        self.publish_image = rospy.get_param('~publish_image', True)

        # Initialize width and height
        self.h = 0
        self.w = 0

        rospy.loginfo("config path: " + self.config_path)
        self.model = load_model(self.config_path, self.weights_path)
        self.model.eval() # Set in evaluation mode
        rospy.loginfo("Deep neural network loaded")

        dims = np.array([1280, 960])
        K = np.array([
          [608.8073, 0.0, 632.53684],
          [0.0, 607.61439, 549.08386],
          [0.0, 0.0, 1.0]])
        dist = 0.998693
        self.undist_ = Undistorter(dims, K, dist)


        # Load classes
        self.classes = load_classes(self.classes_path) # Extracts class labels from file
        self.classes_colors = {}

        # Define subscribers
        self.image_sub = rospy.Subscriber(self.image_topic, ROSImage, self.imageCb, queue_size = 1, buff_size = 2**24)

        # Define publishers
        self.pub_ = rospy.Publisher(self.detected_objects_topic, Detection2DArray, queue_size=10)
        self.pub_viz_ = rospy.Publisher(self.published_image_topic, ROSImage, queue_size=10)
        rospy.loginfo("Launched node for object detection")

    def imageCb(self, data):
        # Convert the image to OpenCV
        self.cv_image = self.undist_.undistort(np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)[:,:,::-1])
        self.h = data.height
        self.w = data.width
        # Initialize detection results
        detections_msg = Detection2DArray()
        detections_msg.header = data.header

        # Configure input
        input_img = self.preprocessImage(self.cv_image)

        # set image type
        if torch.cuda.is_available():
            input_img = input_img.to("cuda")

        # Get detections from network
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections, self.confidence_th, self.nms_th)
            detections = rescale_boxes(detections[0], self.network_img_size, self.cv_image.shape[:2])

        # Parse detections
        if detections[0] is not None:
            for detection in detections:
                # Get xmin, ymin, xmax, ymax, confidence and class
                xmin, ymin, xmax, ymax, conf, det_class = detection

                det_msg = Detection2D()
                det_msg.header = data.header

                result_msg = ObjectHypothesisWithPose()
                result_msg.id = int(det_class)
                result_msg.score = conf
                det_msg.results.append(result_msg)

                # Populate darknet message
                det_msg.bbox.center.x = (xmin+xmax)/2
                det_msg.bbox.center.y = (ymin+ymax)/2
                det_msg.bbox.size_x = (xmax-xmin)
                det_msg.bbox.size_y = (ymax-ymin)


                # Append in overall detection message
                detections_msg.detections.append(det_msg)

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
        return transforms.Compose([
            DEFAULT_TRANSFORMS,
            Resize(self.network_img_size)])(
                (imgIn, np.zeros((1, 5))))[0].unsqueeze(0)

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
    rospy.init_node("detector_manager_node")

    # Define detector object
    dm = DetectorManager()

    # Spin
    rospy.spin()
