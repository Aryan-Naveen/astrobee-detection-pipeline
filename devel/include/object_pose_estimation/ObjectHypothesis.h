// Generated by gencpp from file object_pose_estimation/ObjectHypothesis.msg
// DO NOT EDIT!


#ifndef OBJECT_POSE_ESTIMATION_MESSAGE_OBJECTHYPOTHESIS_H
#define OBJECT_POSE_ESTIMATION_MESSAGE_OBJECTHYPOTHESIS_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace object_pose_estimation
{
template <class ContainerAllocator>
struct ObjectHypothesis_
{
  typedef ObjectHypothesis_<ContainerAllocator> Type;

  ObjectHypothesis_()
    : id(0)
    , score(0.0)  {
    }
  ObjectHypothesis_(const ContainerAllocator& _alloc)
    : id(0)
    , score(0.0)  {
  (void)_alloc;
    }



   typedef int64_t _id_type;
  _id_type id;

   typedef double _score_type;
  _score_type score;





  typedef boost::shared_ptr< ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> const> ConstPtr;

}; // struct ObjectHypothesis_

typedef ::object_pose_estimation::ObjectHypothesis_<std::allocator<void> > ObjectHypothesis;

typedef boost::shared_ptr< ::object_pose_estimation::ObjectHypothesis > ObjectHypothesisPtr;
typedef boost::shared_ptr< ::object_pose_estimation::ObjectHypothesis const> ObjectHypothesisConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> >::stream(s, "", v);
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
struct IsFixedSize< ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> >
{
  static const char* value()
  {
    return "abf73443e563396425a38201e9dacc73";
  }

  static const char* value(const ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xabf73443e5633964ULL;
  static const uint64_t static_value2 = 0x25a38201e9dacc73ULL;
};

template<class ContainerAllocator>
struct DataType< ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> >
{
  static const char* value()
  {
    return "object_pose_estimation/ObjectHypothesis";
  }

  static const char* value(const ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# An object hypothesis that contains no position information.\n\
\n\
# The unique numeric ID of object detected. To get additional information about\n\
#   this ID, such as its human-readable name, listeners should perform a lookup\n\
#   in a metadata database. See vision_msgs/VisionInfo.msg for more detail.\n\
int64 id\n\
\n\
# The probability or confidence value of the detected object. By convention,\n\
#   this value should lie in the range [0-1].\n\
float64 score\n\
";
  }

  static const char* value(const ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.id);
      stream.next(m.score);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ObjectHypothesis_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::object_pose_estimation::ObjectHypothesis_<ContainerAllocator>& v)
  {
    s << indent << "id: ";
    Printer<int64_t>::stream(s, indent + "  ", v.id);
    s << indent << "score: ";
    Printer<double>::stream(s, indent + "  ", v.score);
  }
};

} // namespace message_operations
} // namespace ros

#endif // OBJECT_POSE_ESTIMATION_MESSAGE_OBJECTHYPOTHESIS_H
