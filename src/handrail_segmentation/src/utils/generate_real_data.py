#! /usr/bin/python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
from undistorter import Undistorter

# Instantiate CvBridge
bridge = CvBridge()
undist = Undistorter()
image_id = 0


def image_callback(msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError:
        print("ERROR ")
    else:
        cv2_img = undist.undistort(cv2.resize(cv2_img, (320, 240), interpolation = cv2.INTER_AREA))
        # Save your OpenCV2 image as a jpeg
        image_name = '~/real_data/image_' + str(image_id) + '.png'
        rospy.loginfo('Image ' + image_name + ' success: ' + str(cv2.imwrite(image_name, cv2_img)))
        image_id += 1

def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/hw/cam_dock_color"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
