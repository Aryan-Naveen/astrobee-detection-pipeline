
(cl:in-package :asdf)

(defsystem "handrail_segmentation-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "EkfState" :depends-on ("_package_EkfState"))
    (:file "_package_EkfState" :depends-on ("_package"))
  ))