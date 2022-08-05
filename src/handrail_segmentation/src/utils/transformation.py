import numpy as np
from tf.transformations import *
import tf2_ros
import tf2_py as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import rospy
import pcl, ros_numpy, open3d

class TransformAlignment():
    def __init__(self):
            self.tf_buffer = tf2_ros.Buffer()
            tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def perch_to_dock_transform(self, orig_pointcloud):
        try:
            trans = self.tf_buffer.lookup_transform("dock_cam", orig_pointcloud.header.frame_id,
                                                   orig_pointcloud.header.stamp,
                                                   rospy.Duration(100))
        except tf2.LookupException as ex:
            rospy.logwarn(ex)
            return
        except tf2.ExtrapolationException as ex:
            rospy.logwarn(ex)
            return

        return do_transform_cloud(orig_pointcloud, trans)

    def get_pixel_ids_of_pointcloud(self, pt):
        rospy.loginfo("SIKE")

    def RemapDomainToDomain(self, val,minO,maxO,minN,maxN):
        """Remaps a number in an arbitrary domain to another arbitrary domain"""
        if val>maxO or val<minO: return
        if maxO==minO or maxN==minN: return
        return ((val-minO)/(maxO-minO)*(maxN-minN))+minN

    def Remap2DPointDomain(self, pt,domA,domAA,domB,domBB):
        """Remaps a 2D point from one 2D domain to another (Z unchanged)
        A-->AA; B-->BB. Needs RemapDomainToDomain function"""
        newPtX= self.RemapDomainToDomain(pt[0],domA[0],domA[1],domAA[0],domAA[1])
        newPtY= self.RemapDomainToDomain(pt[1],domB[0],domB[1],domBB[0],domBB[1])
        if newPtX != None and newPtY != None:
            return [newPtX, newPtY]

    def convert_pc_msg_to_np(self, pc_msg):
        # Fix rosbag issues, see: https://github.com/eric-wieser/ros_numpy/issues/23
        offset_sorted = {f.offset: f for f in pc_msg.fields}
        pc_msg.fields = [f for (_, f) in sorted(offset_sorted.items())]

        # Conversion from PointCloud2 msg to np array.
        pc_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc_msg, remove_nans=True)
        return pc_np  # point cloud in numpy and pcl format
