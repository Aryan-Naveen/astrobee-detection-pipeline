// Auto-generated. Do not edit!

// (in-package handrail_segmentation.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class EkfState {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.child_frame_id = null;
      this.pose = null;
      this.velocity = null;
      this.omega = null;
      this.gyro_bias = null;
      this.accel = null;
      this.accel_bias = null;
      this.cov_diag = null;
      this.confidence = null;
      this.aug_state_enum = null;
      this.status = null;
      this.of_count = null;
      this.ml_count = null;
      this.hr_global_pose = null;
      this.ml_mahal_dists = null;
      this.estimating_bias = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('child_frame_id')) {
        this.child_frame_id = initObj.child_frame_id
      }
      else {
        this.child_frame_id = '';
      }
      if (initObj.hasOwnProperty('pose')) {
        this.pose = initObj.pose
      }
      else {
        this.pose = new geometry_msgs.msg.Pose();
      }
      if (initObj.hasOwnProperty('velocity')) {
        this.velocity = initObj.velocity
      }
      else {
        this.velocity = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('omega')) {
        this.omega = initObj.omega
      }
      else {
        this.omega = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('gyro_bias')) {
        this.gyro_bias = initObj.gyro_bias
      }
      else {
        this.gyro_bias = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('accel')) {
        this.accel = initObj.accel
      }
      else {
        this.accel = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('accel_bias')) {
        this.accel_bias = initObj.accel_bias
      }
      else {
        this.accel_bias = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('cov_diag')) {
        this.cov_diag = initObj.cov_diag
      }
      else {
        this.cov_diag = new Array(15).fill(0);
      }
      if (initObj.hasOwnProperty('confidence')) {
        this.confidence = initObj.confidence
      }
      else {
        this.confidence = 0;
      }
      if (initObj.hasOwnProperty('aug_state_enum')) {
        this.aug_state_enum = initObj.aug_state_enum
      }
      else {
        this.aug_state_enum = 0;
      }
      if (initObj.hasOwnProperty('status')) {
        this.status = initObj.status
      }
      else {
        this.status = 0;
      }
      if (initObj.hasOwnProperty('of_count')) {
        this.of_count = initObj.of_count
      }
      else {
        this.of_count = 0;
      }
      if (initObj.hasOwnProperty('ml_count')) {
        this.ml_count = initObj.ml_count
      }
      else {
        this.ml_count = 0;
      }
      if (initObj.hasOwnProperty('hr_global_pose')) {
        this.hr_global_pose = initObj.hr_global_pose
      }
      else {
        this.hr_global_pose = new geometry_msgs.msg.Pose();
      }
      if (initObj.hasOwnProperty('ml_mahal_dists')) {
        this.ml_mahal_dists = initObj.ml_mahal_dists
      }
      else {
        this.ml_mahal_dists = new Array(50).fill(0);
      }
      if (initObj.hasOwnProperty('estimating_bias')) {
        this.estimating_bias = initObj.estimating_bias
      }
      else {
        this.estimating_bias = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type EkfState
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [child_frame_id]
    bufferOffset = _serializer.string(obj.child_frame_id, buffer, bufferOffset);
    // Serialize message field [pose]
    bufferOffset = geometry_msgs.msg.Pose.serialize(obj.pose, buffer, bufferOffset);
    // Serialize message field [velocity]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.velocity, buffer, bufferOffset);
    // Serialize message field [omega]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.omega, buffer, bufferOffset);
    // Serialize message field [gyro_bias]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.gyro_bias, buffer, bufferOffset);
    // Serialize message field [accel]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.accel, buffer, bufferOffset);
    // Serialize message field [accel_bias]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.accel_bias, buffer, bufferOffset);
    // Check that the constant length array field [cov_diag] has the right length
    if (obj.cov_diag.length !== 15) {
      throw new Error('Unable to serialize array field cov_diag - length must be 15')
    }
    // Serialize message field [cov_diag]
    bufferOffset = _arraySerializer.float32(obj.cov_diag, buffer, bufferOffset, 15);
    // Serialize message field [confidence]
    bufferOffset = _serializer.uint8(obj.confidence, buffer, bufferOffset);
    // Serialize message field [aug_state_enum]
    bufferOffset = _serializer.uint8(obj.aug_state_enum, buffer, bufferOffset);
    // Serialize message field [status]
    bufferOffset = _serializer.uint8(obj.status, buffer, bufferOffset);
    // Serialize message field [of_count]
    bufferOffset = _serializer.uint8(obj.of_count, buffer, bufferOffset);
    // Serialize message field [ml_count]
    bufferOffset = _serializer.uint8(obj.ml_count, buffer, bufferOffset);
    // Serialize message field [hr_global_pose]
    bufferOffset = geometry_msgs.msg.Pose.serialize(obj.hr_global_pose, buffer, bufferOffset);
    // Check that the constant length array field [ml_mahal_dists] has the right length
    if (obj.ml_mahal_dists.length !== 50) {
      throw new Error('Unable to serialize array field ml_mahal_dists - length must be 50')
    }
    // Serialize message field [ml_mahal_dists]
    bufferOffset = _arraySerializer.float32(obj.ml_mahal_dists, buffer, bufferOffset, 50);
    // Serialize message field [estimating_bias]
    bufferOffset = _serializer.bool(obj.estimating_bias, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type EkfState
    let len;
    let data = new EkfState(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [child_frame_id]
    data.child_frame_id = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [pose]
    data.pose = geometry_msgs.msg.Pose.deserialize(buffer, bufferOffset);
    // Deserialize message field [velocity]
    data.velocity = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [omega]
    data.omega = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [gyro_bias]
    data.gyro_bias = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [accel]
    data.accel = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [accel_bias]
    data.accel_bias = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [cov_diag]
    data.cov_diag = _arrayDeserializer.float32(buffer, bufferOffset, 15)
    // Deserialize message field [confidence]
    data.confidence = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [aug_state_enum]
    data.aug_state_enum = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [status]
    data.status = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [of_count]
    data.of_count = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [ml_count]
    data.ml_count = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [hr_global_pose]
    data.hr_global_pose = geometry_msgs.msg.Pose.deserialize(buffer, bufferOffset);
    // Deserialize message field [ml_mahal_dists]
    data.ml_mahal_dists = _arrayDeserializer.float32(buffer, bufferOffset, 50)
    // Deserialize message field [estimating_bias]
    data.estimating_bias = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += _getByteLength(object.child_frame_id);
    return length + 502;
  }

  static datatype() {
    // Returns string type for a message object
    return 'handrail_segmentation/EkfState';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '543b97822b033d7199b506ad4005f134';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Copyright (c) 2017, United States Government, as represented by the
    # Administrator of the National Aeronautics and Space Administration.
    #
    # All rights reserved.
    #
    # The Astrobee platform is licensed under the Apache License, Version 2.0
    # (the "License"); you may not use this file except in compliance with the
    # License. You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
    # WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
    # License for the specific language governing permissions and limitations
    # under the License.
    #
    # An observation of a handrail from a depth image.
    
    std_msgs/Header header # header with timestamp
    string child_frame_id # frame ID
    
    geometry_msgs/Pose pose # robot body pose
    
    # m/s
    geometry_msgs/Vector3 velocity # the body velocity
    
    # rad/s
    geometry_msgs/Vector3 omega # body rotational velocity
    geometry_msgs/Vector3 gyro_bias # estimated gyro bias
    
    # m/s/s
    geometry_msgs/Vector3 accel # acceleration in body frame
    geometry_msgs/Vector3 accel_bias # estimated accel bias
    
    # Filter Health
    
    # covariance diagonal. 1-3 orientation, 4-6 gyro bias, 7-9 velocity, 10-12 accel bias, 13-15 position
    float32[15] cov_diag
    # confidence in EKF. 0 is good, 1 is a bit confused, 2 is lost
    uint8 confidence
    uint8 CONFIDENCE_GOOD = 0	# Tracking well
    uint8 CONFIDENCE_POOR = 1	# Tracking poorly
    uint8 CONFIDENCE_LOST = 2	# We are lost
    
    uint8 aug_state_enum # bitmask of augmented states intialized
    
    # status byte sent by GNC
    uint8 status
    uint8 STATUS_INVALID = 255	# invalid
    
    # optical flow features this frame (0 if no update)
    uint8 of_count
    # ml features this frame (0 if no update)
    uint8 ml_count
    
    # Global Handrail Pose
    geometry_msgs/Pose hr_global_pose
    
    # mahalanobis distances for features
    float32[50] ml_mahal_dists
    
    # Are we busy estimating the bias?
    bool estimating_bias
    
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    ================================================================================
    MSG: geometry_msgs/Pose
    # A representation of pose in free space, composed of position and orientation. 
    Point position
    Quaternion orientation
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    ================================================================================
    MSG: geometry_msgs/Quaternion
    # This represents an orientation in free space in quaternion form.
    
    float64 x
    float64 y
    float64 z
    float64 w
    
    ================================================================================
    MSG: geometry_msgs/Vector3
    # This represents a vector in free space. 
    # It is only meant to represent a direction. Therefore, it does not
    # make sense to apply a translation to it (e.g., when applying a 
    # generic rigid transformation to a Vector3, tf2 will only apply the
    # rotation). If you want your data to be translatable too, use the
    # geometry_msgs/Point message instead.
    
    float64 x
    float64 y
    float64 z
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new EkfState(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.child_frame_id !== undefined) {
      resolved.child_frame_id = msg.child_frame_id;
    }
    else {
      resolved.child_frame_id = ''
    }

    if (msg.pose !== undefined) {
      resolved.pose = geometry_msgs.msg.Pose.Resolve(msg.pose)
    }
    else {
      resolved.pose = new geometry_msgs.msg.Pose()
    }

    if (msg.velocity !== undefined) {
      resolved.velocity = geometry_msgs.msg.Vector3.Resolve(msg.velocity)
    }
    else {
      resolved.velocity = new geometry_msgs.msg.Vector3()
    }

    if (msg.omega !== undefined) {
      resolved.omega = geometry_msgs.msg.Vector3.Resolve(msg.omega)
    }
    else {
      resolved.omega = new geometry_msgs.msg.Vector3()
    }

    if (msg.gyro_bias !== undefined) {
      resolved.gyro_bias = geometry_msgs.msg.Vector3.Resolve(msg.gyro_bias)
    }
    else {
      resolved.gyro_bias = new geometry_msgs.msg.Vector3()
    }

    if (msg.accel !== undefined) {
      resolved.accel = geometry_msgs.msg.Vector3.Resolve(msg.accel)
    }
    else {
      resolved.accel = new geometry_msgs.msg.Vector3()
    }

    if (msg.accel_bias !== undefined) {
      resolved.accel_bias = geometry_msgs.msg.Vector3.Resolve(msg.accel_bias)
    }
    else {
      resolved.accel_bias = new geometry_msgs.msg.Vector3()
    }

    if (msg.cov_diag !== undefined) {
      resolved.cov_diag = msg.cov_diag;
    }
    else {
      resolved.cov_diag = new Array(15).fill(0)
    }

    if (msg.confidence !== undefined) {
      resolved.confidence = msg.confidence;
    }
    else {
      resolved.confidence = 0
    }

    if (msg.aug_state_enum !== undefined) {
      resolved.aug_state_enum = msg.aug_state_enum;
    }
    else {
      resolved.aug_state_enum = 0
    }

    if (msg.status !== undefined) {
      resolved.status = msg.status;
    }
    else {
      resolved.status = 0
    }

    if (msg.of_count !== undefined) {
      resolved.of_count = msg.of_count;
    }
    else {
      resolved.of_count = 0
    }

    if (msg.ml_count !== undefined) {
      resolved.ml_count = msg.ml_count;
    }
    else {
      resolved.ml_count = 0
    }

    if (msg.hr_global_pose !== undefined) {
      resolved.hr_global_pose = geometry_msgs.msg.Pose.Resolve(msg.hr_global_pose)
    }
    else {
      resolved.hr_global_pose = new geometry_msgs.msg.Pose()
    }

    if (msg.ml_mahal_dists !== undefined) {
      resolved.ml_mahal_dists = msg.ml_mahal_dists;
    }
    else {
      resolved.ml_mahal_dists = new Array(50).fill(0)
    }

    if (msg.estimating_bias !== undefined) {
      resolved.estimating_bias = msg.estimating_bias;
    }
    else {
      resolved.estimating_bias = false
    }

    return resolved;
    }
};

// Constants for message
EkfState.Constants = {
  CONFIDENCE_GOOD: 0,
  CONFIDENCE_POOR: 1,
  CONFIDENCE_LOST: 2,
  STATUS_INVALID: 255,
}

module.exports = EkfState;
