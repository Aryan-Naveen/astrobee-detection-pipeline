// Generated by gencpp from file object_pose_estimation/BoundingBox2D.msg
// DO NOT EDIT!


#ifndef OBJECT_POSE_ESTIMATION_MESSAGE_BOUNDINGBOX2D_H
#define OBJECT_POSE_ESTIMATION_MESSAGE_BOUNDINGBOX2D_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/Pose2D.h>

namespace object_pose_estimation
{
template <class ContainerAllocator>
struct BoundingBox2D_
{
  typedef BoundingBox2D_<ContainerAllocator> Type;

  BoundingBox2D_()
    : center()
    , size_x(0.0)
    , size_y(0.0)  {
    }
  BoundingBox2D_(const ContainerAllocator& _alloc)
    : center(_alloc)
    , size_x(0.0)
    , size_y(0.0)  {
  (void)_alloc;
    }



   typedef  ::geometry_msgs::Pose2D_<ContainerAllocator>  _center_type;
  _center_type center;

   typedef double _size_x_type;
  _size_x_type size_x;

   typedef double _size_y_type;
  _size_y_type size_y;





  typedef boost::shared_ptr< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> const> ConstPtr;

}; // struct BoundingBox2D_

typedef ::object_pose_estimation::BoundingBox2D_<std::allocator<void> > BoundingBox2D;

typedef boost::shared_ptr< ::object_pose_estimation::BoundingBox2D > BoundingBox2DPtr;
typedef boost::shared_ptr< ::object_pose_estimation::BoundingBox2D const> BoundingBox2DConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace object_pose_estimation

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'object_pose_estimation': ['/home/aryan/Documents/nasa_ws/astrobee-detection-pipeline/src/object_pose_estimation/msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> >
{
  static const char* value()
  {
    return "9ab41e2a4c8627735e5091a9abd68b02";
  }

  static const char* value(const ::object_pose_estimation::BoundingBox2D_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x9ab41e2a4c862773ULL;
  static const uint64_t static_value2 = 0x5e5091a9abd68b02ULL;
};

template<class ContainerAllocator>
struct DataType< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> >
{
  static const char* value()
  {
    return "object_pose_estimation/BoundingBox2D";
  }

  static const char* value(const ::object_pose_estimation::BoundingBox2D_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# A 2D bounding box that can be rotated about its center.\n\
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
";
  }

  static const char* value(const ::object_pose_estimation::BoundingBox2D_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.center);
      stream.next(m.size_x);
      stream.next(m.size_y);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct BoundingBox2D_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::object_pose_estimation::BoundingBox2D_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::object_pose_estimation::BoundingBox2D_<ContainerAllocator>& v)
  {
    s << indent << "center: ";
    s << std::endl;
    Printer< ::geometry_msgs::Pose2D_<ContainerAllocator> >::stream(s, indent + "  ", v.center);
    s << indent << "size_x: ";
    Printer<double>::stream(s, indent + "  ", v.size_x);
    s << indent << "size_y: ";
    Printer<double>::stream(s, indent + "  ", v.size_y);
  }
};

} // namespace message_operations
} // namespace ros

#endif // OBJECT_POSE_ESTIMATION_MESSAGE_BOUNDINGBOX2D_H