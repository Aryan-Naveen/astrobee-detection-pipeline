;; Auto-generated. Do not edit!


(when (boundp 'handrail_segmentation::EkfState)
  (if (not (find-package "HANDRAIL_SEGMENTATION"))
    (make-package "HANDRAIL_SEGMENTATION"))
  (shadow 'EkfState (find-package "HANDRAIL_SEGMENTATION")))
(unless (find-package "HANDRAIL_SEGMENTATION::EKFSTATE")
  (make-package "HANDRAIL_SEGMENTATION::EKFSTATE"))

(in-package "ROS")
;;//! \htmlinclude EkfState.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(intern "*CONFIDENCE_GOOD*" (find-package "HANDRAIL_SEGMENTATION::EKFSTATE"))
(shadow '*CONFIDENCE_GOOD* (find-package "HANDRAIL_SEGMENTATION::EKFSTATE"))
(defconstant handrail_segmentation::EkfState::*CONFIDENCE_GOOD* 0)
(intern "*CONFIDENCE_POOR*" (find-package "HANDRAIL_SEGMENTATION::EKFSTATE"))
(shadow '*CONFIDENCE_POOR* (find-package "HANDRAIL_SEGMENTATION::EKFSTATE"))
(defconstant handrail_segmentation::EkfState::*CONFIDENCE_POOR* 1)
(intern "*CONFIDENCE_LOST*" (find-package "HANDRAIL_SEGMENTATION::EKFSTATE"))
(shadow '*CONFIDENCE_LOST* (find-package "HANDRAIL_SEGMENTATION::EKFSTATE"))
(defconstant handrail_segmentation::EkfState::*CONFIDENCE_LOST* 2)
(intern "*STATUS_INVALID*" (find-package "HANDRAIL_SEGMENTATION::EKFSTATE"))
(shadow '*STATUS_INVALID* (find-package "HANDRAIL_SEGMENTATION::EKFSTATE"))
(defconstant handrail_segmentation::EkfState::*STATUS_INVALID* 255)

(defun handrail_segmentation::EkfState-to-symbol (const)
  (cond
        ((= const 0) 'handrail_segmentation::EkfState::*CONFIDENCE_GOOD*)
        ((= const 1) 'handrail_segmentation::EkfState::*CONFIDENCE_POOR*)
        ((= const 2) 'handrail_segmentation::EkfState::*CONFIDENCE_LOST*)
        ((= const 255) 'handrail_segmentation::EkfState::*STATUS_INVALID*)
        (t nil)))

(defclass handrail_segmentation::EkfState
  :super ros::object
  :slots (_header _child_frame_id _pose _velocity _omega _gyro_bias _accel _accel_bias _cov_diag _confidence _aug_state_enum _status _of_count _ml_count _hr_global_pose _ml_mahal_dists _estimating_bias ))

(defmethod handrail_segmentation::EkfState
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:child_frame_id __child_frame_id) "")
    ((:pose __pose) (instance geometry_msgs::Pose :init))
    ((:velocity __velocity) (instance geometry_msgs::Vector3 :init))
    ((:omega __omega) (instance geometry_msgs::Vector3 :init))
    ((:gyro_bias __gyro_bias) (instance geometry_msgs::Vector3 :init))
    ((:accel __accel) (instance geometry_msgs::Vector3 :init))
    ((:accel_bias __accel_bias) (instance geometry_msgs::Vector3 :init))
    ((:cov_diag __cov_diag) (make-array 15 :initial-element 0.0 :element-type :float))
    ((:confidence __confidence) 0)
    ((:aug_state_enum __aug_state_enum) 0)
    ((:status __status) 0)
    ((:of_count __of_count) 0)
    ((:ml_count __ml_count) 0)
    ((:hr_global_pose __hr_global_pose) (instance geometry_msgs::Pose :init))
    ((:ml_mahal_dists __ml_mahal_dists) (make-array 50 :initial-element 0.0 :element-type :float))
    ((:estimating_bias __estimating_bias) nil)
    )
   (send-super :init)
   (setq _header __header)
   (setq _child_frame_id (string __child_frame_id))
   (setq _pose __pose)
   (setq _velocity __velocity)
   (setq _omega __omega)
   (setq _gyro_bias __gyro_bias)
   (setq _accel __accel)
   (setq _accel_bias __accel_bias)
   (setq _cov_diag __cov_diag)
   (setq _confidence (round __confidence))
   (setq _aug_state_enum (round __aug_state_enum))
   (setq _status (round __status))
   (setq _of_count (round __of_count))
   (setq _ml_count (round __ml_count))
   (setq _hr_global_pose __hr_global_pose)
   (setq _ml_mahal_dists __ml_mahal_dists)
   (setq _estimating_bias __estimating_bias)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:child_frame_id
   (&optional __child_frame_id)
   (if __child_frame_id (setq _child_frame_id __child_frame_id)) _child_frame_id)
  (:pose
   (&rest __pose)
   (if (keywordp (car __pose))
       (send* _pose __pose)
     (progn
       (if __pose (setq _pose (car __pose)))
       _pose)))
  (:velocity
   (&rest __velocity)
   (if (keywordp (car __velocity))
       (send* _velocity __velocity)
     (progn
       (if __velocity (setq _velocity (car __velocity)))
       _velocity)))
  (:omega
   (&rest __omega)
   (if (keywordp (car __omega))
       (send* _omega __omega)
     (progn
       (if __omega (setq _omega (car __omega)))
       _omega)))
  (:gyro_bias
   (&rest __gyro_bias)
   (if (keywordp (car __gyro_bias))
       (send* _gyro_bias __gyro_bias)
     (progn
       (if __gyro_bias (setq _gyro_bias (car __gyro_bias)))
       _gyro_bias)))
  (:accel
   (&rest __accel)
   (if (keywordp (car __accel))
       (send* _accel __accel)
     (progn
       (if __accel (setq _accel (car __accel)))
       _accel)))
  (:accel_bias
   (&rest __accel_bias)
   (if (keywordp (car __accel_bias))
       (send* _accel_bias __accel_bias)
     (progn
       (if __accel_bias (setq _accel_bias (car __accel_bias)))
       _accel_bias)))
  (:cov_diag
   (&optional __cov_diag)
   (if __cov_diag (setq _cov_diag __cov_diag)) _cov_diag)
  (:confidence
   (&optional __confidence)
   (if __confidence (setq _confidence __confidence)) _confidence)
  (:aug_state_enum
   (&optional __aug_state_enum)
   (if __aug_state_enum (setq _aug_state_enum __aug_state_enum)) _aug_state_enum)
  (:status
   (&optional __status)
   (if __status (setq _status __status)) _status)
  (:of_count
   (&optional __of_count)
   (if __of_count (setq _of_count __of_count)) _of_count)
  (:ml_count
   (&optional __ml_count)
   (if __ml_count (setq _ml_count __ml_count)) _ml_count)
  (:hr_global_pose
   (&rest __hr_global_pose)
   (if (keywordp (car __hr_global_pose))
       (send* _hr_global_pose __hr_global_pose)
     (progn
       (if __hr_global_pose (setq _hr_global_pose (car __hr_global_pose)))
       _hr_global_pose)))
  (:ml_mahal_dists
   (&optional __ml_mahal_dists)
   (if __ml_mahal_dists (setq _ml_mahal_dists __ml_mahal_dists)) _ml_mahal_dists)
  (:estimating_bias
   (&optional (__estimating_bias :null))
   (if (not (eq __estimating_bias :null)) (setq _estimating_bias __estimating_bias)) _estimating_bias)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; string _child_frame_id
    4 (length _child_frame_id)
    ;; geometry_msgs/Pose _pose
    (send _pose :serialization-length)
    ;; geometry_msgs/Vector3 _velocity
    (send _velocity :serialization-length)
    ;; geometry_msgs/Vector3 _omega
    (send _omega :serialization-length)
    ;; geometry_msgs/Vector3 _gyro_bias
    (send _gyro_bias :serialization-length)
    ;; geometry_msgs/Vector3 _accel
    (send _accel :serialization-length)
    ;; geometry_msgs/Vector3 _accel_bias
    (send _accel_bias :serialization-length)
    ;; float32[15] _cov_diag
    (* 4    15)
    ;; uint8 _confidence
    1
    ;; uint8 _aug_state_enum
    1
    ;; uint8 _status
    1
    ;; uint8 _of_count
    1
    ;; uint8 _ml_count
    1
    ;; geometry_msgs/Pose _hr_global_pose
    (send _hr_global_pose :serialization-length)
    ;; float32[50] _ml_mahal_dists
    (* 4    50)
    ;; bool _estimating_bias
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; string _child_frame_id
       (write-long (length _child_frame_id) s) (princ _child_frame_id s)
     ;; geometry_msgs/Pose _pose
       (send _pose :serialize s)
     ;; geometry_msgs/Vector3 _velocity
       (send _velocity :serialize s)
     ;; geometry_msgs/Vector3 _omega
       (send _omega :serialize s)
     ;; geometry_msgs/Vector3 _gyro_bias
       (send _gyro_bias :serialize s)
     ;; geometry_msgs/Vector3 _accel
       (send _accel :serialize s)
     ;; geometry_msgs/Vector3 _accel_bias
       (send _accel_bias :serialize s)
     ;; float32[15] _cov_diag
     (dotimes (i 15)
       (sys::poke (elt _cov_diag i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; uint8 _confidence
       (write-byte _confidence s)
     ;; uint8 _aug_state_enum
       (write-byte _aug_state_enum s)
     ;; uint8 _status
       (write-byte _status s)
     ;; uint8 _of_count
       (write-byte _of_count s)
     ;; uint8 _ml_count
       (write-byte _ml_count s)
     ;; geometry_msgs/Pose _hr_global_pose
       (send _hr_global_pose :serialize s)
     ;; float32[50] _ml_mahal_dists
     (dotimes (i 50)
       (sys::poke (elt _ml_mahal_dists i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; bool _estimating_bias
       (if _estimating_bias (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; string _child_frame_id
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _child_frame_id (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; geometry_msgs/Pose _pose
     (send _pose :deserialize buf ptr-) (incf ptr- (send _pose :serialization-length))
   ;; geometry_msgs/Vector3 _velocity
     (send _velocity :deserialize buf ptr-) (incf ptr- (send _velocity :serialization-length))
   ;; geometry_msgs/Vector3 _omega
     (send _omega :deserialize buf ptr-) (incf ptr- (send _omega :serialization-length))
   ;; geometry_msgs/Vector3 _gyro_bias
     (send _gyro_bias :deserialize buf ptr-) (incf ptr- (send _gyro_bias :serialization-length))
   ;; geometry_msgs/Vector3 _accel
     (send _accel :deserialize buf ptr-) (incf ptr- (send _accel :serialization-length))
   ;; geometry_msgs/Vector3 _accel_bias
     (send _accel_bias :deserialize buf ptr-) (incf ptr- (send _accel_bias :serialization-length))
   ;; float32[15] _cov_diag
   (dotimes (i (length _cov_diag))
     (setf (elt _cov_diag i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; uint8 _confidence
     (setq _confidence (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _aug_state_enum
     (setq _aug_state_enum (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _status
     (setq _status (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _of_count
     (setq _of_count (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _ml_count
     (setq _ml_count (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; geometry_msgs/Pose _hr_global_pose
     (send _hr_global_pose :deserialize buf ptr-) (incf ptr- (send _hr_global_pose :serialization-length))
   ;; float32[50] _ml_mahal_dists
   (dotimes (i (length _ml_mahal_dists))
     (setf (elt _ml_mahal_dists i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; bool _estimating_bias
     (setq _estimating_bias (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(setf (get handrail_segmentation::EkfState :md5sum-) "543b97822b033d7199b506ad4005f134")
(setf (get handrail_segmentation::EkfState :datatype-) "handrail_segmentation/EkfState")
(setf (get handrail_segmentation::EkfState :definition-)
      "# Copyright (c) 2017, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
#
# All rights reserved.
#
# The Astrobee platform is licensed under the Apache License, Version 2.0
# (the \"License\"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS, WITHOUT
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
")



(provide :handrail_segmentation/EkfState "543b97822b033d7199b506ad4005f134")


