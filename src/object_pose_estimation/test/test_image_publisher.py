#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import numpy as np
from sensor_msgs.msg import Image as ROSImage

from rospkg import RosPack
package = RosPack()
package_path = package.get_path('object_pose_estimation')

class TestingImagePublisher(object):
    def __init__(self):
        # Params
        self.image = None
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(1)

        # Publishers
        self.pub = rospy.Publisher('sample_img', Image,queue_size=10)
        self.img_path = os.path.join(package_path, 'test/example.png')

    def start(self):
        rospy.loginfo('Starting publishing image')
        while not rospy.is_shutdown():
            self.image = cv2.imread(self.img_path)
            if self.image is not None:
                self.pub.publish(self.cv2_to_imgmsg(self.image))
            self.loop_rate.sleep()


    def cv2_to_imgmsg(self, img):
        new_img_msg = ROSImage()
        new_img_msg.height = img.shape[0]
        new_img_msg.width = img.shape[1]
        new_img_msg.encoding = 'bgr8'
        new_img_msg.step = new_img_msg.width*3
        new_img_msg.data = np.array(img)[:,:,::-1].tobytes()
        return new_img_msg


if __name__ == '__main__':
    rospy.init_node("sample_img_test_node", anonymous=True)
    my_node = TestingImagePublisher()
    my_node.start()
