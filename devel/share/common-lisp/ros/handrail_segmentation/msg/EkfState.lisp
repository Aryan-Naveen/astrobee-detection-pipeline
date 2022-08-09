; Auto-generated. Do not edit!


(cl:in-package handrail_segmentation-msg)


;//! \htmlinclude EkfState.msg.html

(cl:defclass <EkfState> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (child_frame_id
    :reader child_frame_id
    :initarg :child_frame_id
    :type cl:string
    :initform "")
   (pose
    :reader pose
    :initarg :pose
    :type geometry_msgs-msg:Pose
    :initform (cl:make-instance 'geometry_msgs-msg:Pose))
   (velocity
    :reader velocity
    :initarg :velocity
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (omega
    :reader omega
    :initarg :omega
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (gyro_bias
    :reader gyro_bias
    :initarg :gyro_bias
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (accel
    :reader accel
    :initarg :accel
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (accel_bias
    :reader accel_bias
    :initarg :accel_bias
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (cov_diag
    :reader cov_diag
    :initarg :cov_diag
    :type (cl:vector cl:float)
   :initform (cl:make-array 15 :element-type 'cl:float :initial-element 0.0))
   (confidence
    :reader confidence
    :initarg :confidence
    :type cl:fixnum
    :initform 0)
   (aug_state_enum
    :reader aug_state_enum
    :initarg :aug_state_enum
    :type cl:fixnum
    :initform 0)
   (status
    :reader status
    :initarg :status
    :type cl:fixnum
    :initform 0)
   (of_count
    :reader of_count
    :initarg :of_count
    :type cl:fixnum
    :initform 0)
   (ml_count
    :reader ml_count
    :initarg :ml_count
    :type cl:fixnum
    :initform 0)
   (hr_global_pose
    :reader hr_global_pose
    :initarg :hr_global_pose
    :type geometry_msgs-msg:Pose
    :initform (cl:make-instance 'geometry_msgs-msg:Pose))
   (ml_mahal_dists
    :reader ml_mahal_dists
    :initarg :ml_mahal_dists
    :type (cl:vector cl:float)
   :initform (cl:make-array 50 :element-type 'cl:float :initial-element 0.0))
   (estimating_bias
    :reader estimating_bias
    :initarg :estimating_bias
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass EkfState (<EkfState>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <EkfState>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'EkfState)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name handrail_segmentation-msg:<EkfState> is deprecated: use handrail_segmentation-msg:EkfState instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:header-val is deprecated.  Use handrail_segmentation-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'child_frame_id-val :lambda-list '(m))
(cl:defmethod child_frame_id-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:child_frame_id-val is deprecated.  Use handrail_segmentation-msg:child_frame_id instead.")
  (child_frame_id m))

(cl:ensure-generic-function 'pose-val :lambda-list '(m))
(cl:defmethod pose-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:pose-val is deprecated.  Use handrail_segmentation-msg:pose instead.")
  (pose m))

(cl:ensure-generic-function 'velocity-val :lambda-list '(m))
(cl:defmethod velocity-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:velocity-val is deprecated.  Use handrail_segmentation-msg:velocity instead.")
  (velocity m))

(cl:ensure-generic-function 'omega-val :lambda-list '(m))
(cl:defmethod omega-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:omega-val is deprecated.  Use handrail_segmentation-msg:omega instead.")
  (omega m))

(cl:ensure-generic-function 'gyro_bias-val :lambda-list '(m))
(cl:defmethod gyro_bias-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:gyro_bias-val is deprecated.  Use handrail_segmentation-msg:gyro_bias instead.")
  (gyro_bias m))

(cl:ensure-generic-function 'accel-val :lambda-list '(m))
(cl:defmethod accel-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:accel-val is deprecated.  Use handrail_segmentation-msg:accel instead.")
  (accel m))

(cl:ensure-generic-function 'accel_bias-val :lambda-list '(m))
(cl:defmethod accel_bias-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:accel_bias-val is deprecated.  Use handrail_segmentation-msg:accel_bias instead.")
  (accel_bias m))

(cl:ensure-generic-function 'cov_diag-val :lambda-list '(m))
(cl:defmethod cov_diag-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:cov_diag-val is deprecated.  Use handrail_segmentation-msg:cov_diag instead.")
  (cov_diag m))

(cl:ensure-generic-function 'confidence-val :lambda-list '(m))
(cl:defmethod confidence-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:confidence-val is deprecated.  Use handrail_segmentation-msg:confidence instead.")
  (confidence m))

(cl:ensure-generic-function 'aug_state_enum-val :lambda-list '(m))
(cl:defmethod aug_state_enum-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:aug_state_enum-val is deprecated.  Use handrail_segmentation-msg:aug_state_enum instead.")
  (aug_state_enum m))

(cl:ensure-generic-function 'status-val :lambda-list '(m))
(cl:defmethod status-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:status-val is deprecated.  Use handrail_segmentation-msg:status instead.")
  (status m))

(cl:ensure-generic-function 'of_count-val :lambda-list '(m))
(cl:defmethod of_count-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:of_count-val is deprecated.  Use handrail_segmentation-msg:of_count instead.")
  (of_count m))

(cl:ensure-generic-function 'ml_count-val :lambda-list '(m))
(cl:defmethod ml_count-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:ml_count-val is deprecated.  Use handrail_segmentation-msg:ml_count instead.")
  (ml_count m))

(cl:ensure-generic-function 'hr_global_pose-val :lambda-list '(m))
(cl:defmethod hr_global_pose-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:hr_global_pose-val is deprecated.  Use handrail_segmentation-msg:hr_global_pose instead.")
  (hr_global_pose m))

(cl:ensure-generic-function 'ml_mahal_dists-val :lambda-list '(m))
(cl:defmethod ml_mahal_dists-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:ml_mahal_dists-val is deprecated.  Use handrail_segmentation-msg:ml_mahal_dists instead.")
  (ml_mahal_dists m))

(cl:ensure-generic-function 'estimating_bias-val :lambda-list '(m))
(cl:defmethod estimating_bias-val ((m <EkfState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader handrail_segmentation-msg:estimating_bias-val is deprecated.  Use handrail_segmentation-msg:estimating_bias instead.")
  (estimating_bias m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<EkfState>)))
    "Constants for message type '<EkfState>"
  '((:CONFIDENCE_GOOD . 0)
    (:CONFIDENCE_POOR . 1)
    (:CONFIDENCE_LOST . 2)
    (:STATUS_INVALID . 255))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'EkfState)))
    "Constants for message type 'EkfState"
  '((:CONFIDENCE_GOOD . 0)
    (:CONFIDENCE_POOR . 1)
    (:CONFIDENCE_LOST . 2)
    (:STATUS_INVALID . 255))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <EkfState>) ostream)
  "Serializes a message object of type '<EkfState>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'child_frame_id))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'child_frame_id))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'pose) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'velocity) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'omega) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'gyro_bias) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'accel) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'accel_bias) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'cov_diag))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'confidence)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'aug_state_enum)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'status)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'of_count)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'ml_count)) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'hr_global_pose) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'ml_mahal_dists))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'estimating_bias) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <EkfState>) istream)
  "Deserializes a message object of type '<EkfState>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'child_frame_id) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'child_frame_id) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'pose) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'velocity) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'omega) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'gyro_bias) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'accel) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'accel_bias) istream)
  (cl:setf (cl:slot-value msg 'cov_diag) (cl:make-array 15))
  (cl:let ((vals (cl:slot-value msg 'cov_diag)))
    (cl:dotimes (i 15)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'confidence)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'aug_state_enum)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'status)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'of_count)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'ml_count)) (cl:read-byte istream))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'hr_global_pose) istream)
  (cl:setf (cl:slot-value msg 'ml_mahal_dists) (cl:make-array 50))
  (cl:let ((vals (cl:slot-value msg 'ml_mahal_dists)))
    (cl:dotimes (i 50)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
    (cl:setf (cl:slot-value msg 'estimating_bias) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<EkfState>)))
  "Returns string type for a message object of type '<EkfState>"
  "handrail_segmentation/EkfState")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'EkfState)))
  "Returns string type for a message object of type 'EkfState"
  "handrail_segmentation/EkfState")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<EkfState>)))
  "Returns md5sum for a message object of type '<EkfState>"
  "543b97822b033d7199b506ad4005f134")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'EkfState)))
  "Returns md5sum for a message object of type 'EkfState"
  "543b97822b033d7199b506ad4005f134")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<EkfState>)))
  "Returns full string definition for message of type '<EkfState>"
  (cl:format cl:nil "# Copyright (c) 2017, United States Government, as represented by the~%# Administrator of the National Aeronautics and Space Administration.~%#~%# All rights reserved.~%#~%# The Astrobee platform is licensed under the Apache License, Version 2.0~%# (the \"License\"); you may not use this file except in compliance with the~%# License. You may obtain a copy of the License at~%#~%#     http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS, WITHOUT~%# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the~%# License for the specific language governing permissions and limitations~%# under the License.~%#~%# An observation of a handrail from a depth image.~%~%std_msgs/Header header # header with timestamp~%string child_frame_id # frame ID~%~%geometry_msgs/Pose pose # robot body pose~%~%# m/s~%geometry_msgs/Vector3 velocity # the body velocity~%~%# rad/s~%geometry_msgs/Vector3 omega # body rotational velocity~%geometry_msgs/Vector3 gyro_bias # estimated gyro bias~%~%# m/s/s~%geometry_msgs/Vector3 accel # acceleration in body frame~%geometry_msgs/Vector3 accel_bias # estimated accel bias~%~%# Filter Health~%~%# covariance diagonal. 1-3 orientation, 4-6 gyro bias, 7-9 velocity, 10-12 accel bias, 13-15 position~%float32[15] cov_diag~%# confidence in EKF. 0 is good, 1 is a bit confused, 2 is lost~%uint8 confidence~%uint8 CONFIDENCE_GOOD = 0	# Tracking well~%uint8 CONFIDENCE_POOR = 1	# Tracking poorly~%uint8 CONFIDENCE_LOST = 2	# We are lost~%~%uint8 aug_state_enum # bitmask of augmented states intialized~%~%# status byte sent by GNC~%uint8 status~%uint8 STATUS_INVALID = 255	# invalid~%~%# optical flow features this frame (0 if no update)~%uint8 of_count~%# ml features this frame (0 if no update)~%uint8 ml_count~%~%# Global Handrail Pose~%geometry_msgs/Pose hr_global_pose~%~%# mahalanobis distances for features~%float32[50] ml_mahal_dists~%~%# Are we busy estimating the bias?~%bool estimating_bias~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'EkfState)))
  "Returns full string definition for message of type 'EkfState"
  (cl:format cl:nil "# Copyright (c) 2017, United States Government, as represented by the~%# Administrator of the National Aeronautics and Space Administration.~%#~%# All rights reserved.~%#~%# The Astrobee platform is licensed under the Apache License, Version 2.0~%# (the \"License\"); you may not use this file except in compliance with the~%# License. You may obtain a copy of the License at~%#~%#     http://www.apache.org/licenses/LICENSE-2.0~%#~%# Unless required by applicable law or agreed to in writing, software~%# distributed under the License is distributed on an \"AS IS\" BASIS, WITHOUT~%# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the~%# License for the specific language governing permissions and limitations~%# under the License.~%#~%# An observation of a handrail from a depth image.~%~%std_msgs/Header header # header with timestamp~%string child_frame_id # frame ID~%~%geometry_msgs/Pose pose # robot body pose~%~%# m/s~%geometry_msgs/Vector3 velocity # the body velocity~%~%# rad/s~%geometry_msgs/Vector3 omega # body rotational velocity~%geometry_msgs/Vector3 gyro_bias # estimated gyro bias~%~%# m/s/s~%geometry_msgs/Vector3 accel # acceleration in body frame~%geometry_msgs/Vector3 accel_bias # estimated accel bias~%~%# Filter Health~%~%# covariance diagonal. 1-3 orientation, 4-6 gyro bias, 7-9 velocity, 10-12 accel bias, 13-15 position~%float32[15] cov_diag~%# confidence in EKF. 0 is good, 1 is a bit confused, 2 is lost~%uint8 confidence~%uint8 CONFIDENCE_GOOD = 0	# Tracking well~%uint8 CONFIDENCE_POOR = 1	# Tracking poorly~%uint8 CONFIDENCE_LOST = 2	# We are lost~%~%uint8 aug_state_enum # bitmask of augmented states intialized~%~%# status byte sent by GNC~%uint8 status~%uint8 STATUS_INVALID = 255	# invalid~%~%# optical flow features this frame (0 if no update)~%uint8 of_count~%# ml features this frame (0 if no update)~%uint8 ml_count~%~%# Global Handrail Pose~%geometry_msgs/Pose hr_global_pose~%~%# mahalanobis distances for features~%float32[50] ml_mahal_dists~%~%# Are we busy estimating the bias?~%bool estimating_bias~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <EkfState>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:length (cl:slot-value msg 'child_frame_id))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'pose))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'velocity))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'omega))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'gyro_bias))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'accel))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'accel_bias))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'cov_diag) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     1
     1
     1
     1
     1
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'hr_global_pose))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'ml_mahal_dists) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <EkfState>))
  "Converts a ROS message object to a list"
  (cl:list 'EkfState
    (cl:cons ':header (header msg))
    (cl:cons ':child_frame_id (child_frame_id msg))
    (cl:cons ':pose (pose msg))
    (cl:cons ':velocity (velocity msg))
    (cl:cons ':omega (omega msg))
    (cl:cons ':gyro_bias (gyro_bias msg))
    (cl:cons ':accel (accel msg))
    (cl:cons ':accel_bias (accel_bias msg))
    (cl:cons ':cov_diag (cov_diag msg))
    (cl:cons ':confidence (confidence msg))
    (cl:cons ':aug_state_enum (aug_state_enum msg))
    (cl:cons ':status (status msg))
    (cl:cons ':of_count (of_count msg))
    (cl:cons ':ml_count (ml_count msg))
    (cl:cons ':hr_global_pose (hr_global_pose msg))
    (cl:cons ':ml_mahal_dists (ml_mahal_dists msg))
    (cl:cons ':estimating_bias (estimating_bias msg))
))
