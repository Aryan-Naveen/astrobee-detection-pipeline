// Generated by gencpp from file object_pose_estimation/Detection2D.msg
// DO NOT EDIT!


#ifndef OBJECT_POSE_ESTIMATION_MESSAGE_DETECTION2D_H
#define OBJECT_POSE_ESTIMATION_MESSAGE_DETECTION2D_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <object_pose_estimation/ObjectHypothesisWithPose.h>
#include <object_pose_estimation/BoundingBox2D.h>
#include <sensor_msgs/Image.h>

namespace object_pose_estimation
{
template <class ContainerAllocator>
struct Detection2D_
{
  typedef Detection2D_<ContainerAllocator> Type;

  Detection2D_()
    : header()
    , results()
    , bbox()
    , source_img()  {
    }
  Detection2D_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , results(_alloc)
    , bbox(_alloc)
    , source_img(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::vector< ::object_pose_estimation::ObjectHypothesisWithPose_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::object_pose_estimation::ObjectHypothesisWithPose_<ContainerAllocator> >::other >  _results_type;
  _results_type results;

   typedef  ::object_pose_estimation::BoundingBox2D_<ContainerAllocator>  _bbox_type;
  _bbox_type bbox;

   typedef  ::sensor_msgs::Image_<ContainerAllocator>  _source_img_type;
  _source_img_type source_img;





  typedef boost::shared_ptr< ::object_pose_estimation::Detection2D_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::object_pose_estimation::Detection2D_<ContainerAllocator> const> ConstPtr;

}; // struct Detection2D_

typedef ::object_pose_estimation::Detection2D_<std::allocator<void> > Detection2D;

typedef boost::shared_ptr< ::object_pose_estimation::Detection2D > Detection2DPtr;
typedef boost::shared_ptr< ::object_pose_estimation::Detection2D const> Detection2DConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::object_pose_estimation::Detection2D_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::object_pose_estimation::Detection2D_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace object_pose_estimation

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'object_pose_estimation': ['/home/aryan/Documents/nasa_ws/astrobee-detection-pipeline/src/object_pose_estimation/msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::object_pose_estimation::Detection2D_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::object_pose_estimation::Detection2D_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::object_pose_estimation::Detection2D_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::object_pose_estimation::Detection2D_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::object_pose_estimation::Detection2D_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::object_pose_estimation::Detection2D_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::object_pose_estimation::Detection2D_<ContainerAllocator> >
{
  static const char* value()
  {
    return "9e11092151fa150724a255fbac727f3b";
  }

  static const char* value(const ::object_pose_estimation::Detection2D_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x9e11092151fa1507ULL;
  static const uint64_t static_value2 = 0x24a255fbac727f3bULL;
};

template<class ContainerAllocator>
struct DataType< ::object_pose_estimation::Detection2D_<ContainerAllocator> >
{
  static const char* value()
  {
    return "object_pose_estimation/Detection2D";
  }

  static const char* value(const ::object_pose_estimation::Detection2D_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::object_pose_estimation::Detection2D_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Defines a 2D detection result.\n\
#\n\
# This is similar to a 2D classification, but includes position information,\n\
#   allowing a classification result for a specific crop or image point to\n\
#   to be located in the larger image.\n\
\n\
Header header\n\
\n\
# Class probabilities\n\
ObjectHypothesisWithPose[] results\n\
\n\
# 2D bounding box surrounding the object.\n\
BoundingBox2D bbox\n\
\n\
# The 2D data that generated these results (i.e. region proposal cropped out of\n\
#   the image). Not required for all use cases, so it may be empty.\n\
sensor_msgs/Image source_img\n\
\n\
================================================================================\n\
MSG: std_msgs/Header\n\
# Standard metadata for higher-level stamped data types.\n\
# This is generally used to communicate timestamped data \n\
# in a particular coordinate frame.\n\
# \n\
# sequence ID: consecutively increasing ID \n\
uint32 seq\n\
#Two-integer timestamp that is expressed as:\n\
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n\
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n\
# time-handling sugar is provided by the client library\n\
time stamp\n\
#Frame this data is associated with\n\
# 0: no frame\n\
# 1: global frame\n\
string frame_id\n\
\n\
================================================================================\n\
MSG: object_pose_estimation/ObjectHypothesisWithPose\n\
# An object hypothesis that contains position information.\n\
\n\
# The unique numeric ID of object detected. To get additional information about\n\
#   this ID, such as its human-readable name, listeners should perform a lookup\n\
#   in a metadata database. See vision_msgs/VisionInfo.msg for more detail.\n\
int64 id\n\
\n\
# The probability or confidence value of the detected object. By convention,\n\
#   this value should lie in the range [0-1].\n\
float64 score\n\
\n\
# The 6D pose of the object hypothesis. This pose should be\n\
#   defined as the pose of some fixed reference point on the object, such a\n\
#   the geometric center of the bounding box or the center of mass of the\n\
#   object.\n\
# Note that this pose is not stamped; frame information can be defined by\n\
#   parent messages.\n\
# Also note that different classes predicted for the same input data may have\n\
#   different predicted 6D poses.\n\
geometry_msgs/PoseWithCovariance pose\n\
================================================================================\n\
MSG: geometry_msgs/PoseWithCovariance\n\
# This represents a pose in free space with uncertainty.\n\
\n\
Pose pose\n\
\n\
# Row-major representation of the 6x6 covariance matrix\n\
# The orientation parameters use a fixed-axis representation.\n\
# In order, the parameters are:\n\
# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)\n\
float64[36] covariance\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Pose\n\
# A representation of pose in free space, composed of position and orientation. \n\
Point position\n\
Quaternion orientation\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Point\n\
# This contains the position of a point in free space\n\
float64 x\n\
float64 y\n\
float64 z\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Quaternion\n\
# This represents an orientation in free space in quaternion form.\n\
\n\
float64 x\n\
float64 y\n\
float64 z\n\
float64 w\n\
\n\
================================================================================\n\
MSG: object_pose_estimation/BoundingBox2D\n\
# A 2D bounding box that can be rotated about its center.\n\
# All dimensions are in pixels, but represented using floating-point\n\
#   values to allow sub-pixel precision. If an exact pixel crop is required\n\
#   for a rotated bounding box, it can be calculated using Bresenham's line\n\
#   algorithm.\n\
\n\
# The 2D position (in pixels) and orientation of the bounding box center.\n\
geometry_msgs/Pose2D center\n\
\n\
# The size (in pixels) of the bounding box surrounding the object relative\n\
#   to the pose of its center.\n\
float64 size_x\n\
float64 size_y\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Pose2D\n\
# Deprecated\n\
# Please use the full 3D pose.\n\
\n\
# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.\n\
\n\
# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.\n\
\n\
\n\
# This expresses a position and orientation on a 2D manifold.\n\
\n\
float64 x\n\
float64 y\n\
float64 theta\n\
\n\
================================================================================\n\
MSG: sensor_msgs/Image\n\
# This message contains an uncompressed image\n\
# (0, 0) is at top-left corner of image\n\
#\n\
\n\
Header header        # Header timestamp should be acquisition time of image\n\
                     # Header frame_id should be optical frame of camera\n\
                     # origin of frame should be optical center of camera\n\
                     # +x should point to the right in the image\n\
                     # +y should point down in the image\n\
                     # +z should point into to plane of the image\n\
                     # If the frame_id here and the frame_id of the CameraInfo\n\
                     # message associated with the image conflict\n\
                     # the behavior is undefined\n\
\n\
uint32 height         # image height, that is, number of rows\n\
uint32 width          # image width, that is, number of columns\n\
\n\
# The legal values for encoding are in file src/image_encodings.cpp\n\
# If you want to standardize a new string format, join\n\
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.\n\
\n\
string encoding       # Encoding of pixels -- channel meaning, ordering, size\n\
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h\n\
\n\
uint8 is_bigendian    # is this data bigendian?\n\
uint32 step           # Full row length in bytes\n\
uint8[] data          # actual matrix data, size is (step * rows)\n\
";
  }

  static const char* value(const ::object_pose_estimation::Detection2D_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::object_pose_estimation::Detection2D_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.results);
      stream.next(m.bbox);
      stream.next(m.source_img);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Detection2D_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::object_pose_estimation::Detection2D_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::object_pose_estimation::Detection2D_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "results[]" << std::endl;
    for (size_t i = 0; i < v.results.size(); ++i)
    {
      s << indent << "  results[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::object_pose_estimation::ObjectHypothesisWithPose_<ContainerAllocator> >::stream(s, indent + "    ", v.results[i]);
    }
    s << indent << "bbox: ";
    s << std::endl;
    Printer< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> >::stream(s, indent + "  ", v.bbox);
    s << indent << "source_img: ";
    s << std::endl;
    Printer< ::sensor_msgs::Image_<ContainerAllocator> >::stream(s, indent + "  ", v.source_img);
  }
};

} // namespace message_operations
} // namespace ros

#endif // OBJECT_POSE_ESTIMATION_MESSAGE_DETECTION2D_H
