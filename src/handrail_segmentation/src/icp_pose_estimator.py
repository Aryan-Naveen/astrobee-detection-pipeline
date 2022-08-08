#!/usr/bin/env python3

from __future__ import division

import rospy
from rospkg import RosPack
from std_msgs.msg import UInt8, Int16MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2
import utils.icp

import open3d as o3d
import numpy as np
from utils.transformation import TransformAlignment
import utils.icp as icp
from utils.converter import *
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.voxel_down_sample(voxel_size=0.1)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # o3d.visualization.draw_geometries([source_temp, target_temp], point_show_normal=True)


def preprocess_point_cloud(pcd, voxel_size, radius, should_voxel=True):
    if should_voxel:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    else:
        pcd_down = pcd
    radius_normal = radius * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = radius * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh





def execute_global_registration(src, target, voxel_size=0.01):

    source_down, source_fpfh = preprocess_point_cloud(src, voxel_size, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size, 75*voxel_size)

    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    draw_registration_result(source_down, target_down, result.transformation)
    return result


class DOFPoseEstimator():
    def __init__(self):
        self.pointcloud_topic = rospy.get_param('~segmented_object_topic', '/hw/detected_handrail')
        self.pub_ = rospy.Publisher('/hw/transformed/evaluate', PointCloud2, queue_size=10)

        # Define subscribers
        self.pointcloud_sub = rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.pointcloud_callback, queue_size = 1, buff_size = 2**24)
        self.align_transformer = TransformAlignment()

    def pointcloud_callback(self, points_msg):
        detected_handrail = self.align_transformer.convert_pc_msg_to_np(points_msg)
        detected_handrail_o3d = o3d.geometry.PointCloud()
        detected_handrail_o3d.points = o3d.utility.Vector3dVector(detected_handrail)



        registered_handrail_o3d = o3d.io.read_point_cloud('/home/anaveen/Documents/nasa_ws/astrobee-detection-pipeline/src/handrail_segmentation/src/reference_pointclouds/handrail_30.pcd')
        registered_handrail = np.asarray(registered_handrail_o3d.points)

        rospy.loginfo("Running ICP to estimate 6 DOF pose of detected object")
        reg_p2p = execute_global_registration(registered_handrail_o3d, detected_handrail_o3d)
        rospy.loginfo("Done running ICP to estimate 6 DOF pose of detected object")

        T = np.asarray(reg_p2p.transformation)

        # visualize = []
        # for point in registered_handrail:
        #     visualize.append(np.dot(T, np.append(point.reshape(3, ), 1))[:3])
        #
        # self.pub_.publish(convertPc2(visualize))
        rospy.loginfo(T)
        rospy.loginfo("------------------Transformation computed----------------")


if __name__=="__main__":
    # Initialize node
    rospy.init_node("icp_pose_estimator_node")

    # Define detector object
    dm = DOFPoseEstimator()

    # Spin
    rospy.spin()
