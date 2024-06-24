#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2024 Marc Bestmann <marc.bestmann@dlr.de>
#
# SPDX-License-Identifier: MIT

"""This Module implements an interactive marker which is used to select datapoints."""
import sys

from geometry_msgs.msg import Pose
from interactive_markers import InteractiveMarkerServer
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, \
    InteractiveMarkerFeedback, Marker


class PoseMarker(Node):
    """Handles the interactive marker to select data points."""

    def __init__(self):
        """
        Initialize the interactive marker.

        :param node: ROS node
        :param inspection: Inspection object
        :return: None
        """
        super(PoseMarker, self).__init__('pose_marker')
        self.server = InteractiveMarkerServer(self, 'pose_marker')
        # 0.141505 0.9883 -1.38115 95.4226 1.953 269.771
        self.pose = Pose()
        self.pose.position.x = 0.0
        self.pose.position.y = 0.0
        self.pose.position.z = 0.0
        self.pose.orientation.w = 0.0
        self.pose.orientation.x = 0.0
        self.pose.orientation.y = 0.0
        self.pose.orientation.z = 1.0
        self.create_arrow_marker(
            InteractiveMarkerControl.MOVE_ROTATE_3D, self.pose)
        self.server.applyChanges()

    def process_arrow_feedback(self, feedback: InteractiveMarkerFeedback) -> None:
        """
        Used to set the new pose of the marker.

        :return: None
        """
        self.pose = feedback.pose

    def create_arrow(self) -> Marker:
        """
        Is called to create a marker.arrow object.

        :return: Marker.arrow object
        """
        marker = Marker()

        marker.type = Marker.ARROW
        marker.header.frame_id = 'world'
        marker.scale.x = 0.1
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.1
        marker.color.b = 0.1
        marker.color.a = 1.0
        marker.pose.position.x = -0.05
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        return marker

    def create_arrow_control(self) -> InteractiveMarkerControl:
        """
        Is called to set the Control object for the interactice marker.

        :return: control obejct
        """
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self.create_arrow())
        return control

    def normalize_quaternion(self, quaternion_msg):
        """
        Normalize the quaternion to make sure it is a unit quaternion
        :param quaternion_msg: quaternion message containing the to be normalied quaternion
        :return: normalized quaternion
        """
        norm = quaternion_msg.x**2 + quaternion_msg.y**2 + \
            quaternion_msg.z**2 + quaternion_msg.w**2
        s = norm ** (-0.5)
        quaternion_msg.x *= s
        quaternion_msg.y *= s
        quaternion_msg.z *= s
        quaternion_msg.w *= s

    def create_arrow_marker(self, interaction_mode: InteractiveMarkerControl, pose: Pose) \
            -> None:
        """
        Is called to create the interactive marker and to give this object to the interactive \
        marker server.

        :param interaction_mode: way of controling the object. Either MOVE_3D or MOVE_ROTATE_3D
        :param position: starting postion
        :return: None
        """
        int_marker = InteractiveMarker()
        # todo maybe make this a parameter
        int_marker.header.frame_id = 'world'
        int_marker.pose = pose

        int_marker.name = 'simple_3dof'
        int_marker.description = 'Simple 3-DOF Control'

        # insert an arrow
        int_marker.controls.append(self.create_arrow_control())
        int_marker.controls[0].interaction_mode = interaction_mode
        int_marker.scale = 0.15

        if interaction_mode != InteractiveMarkerControl.NONE:
            control_modes_dict = {
                InteractiveMarkerControl.MOVE_3D: 'MOVE_3D',
                InteractiveMarkerControl.MOVE_ROTATE_3D: 'MOVE_ROTATE_3D',
            }
            int_marker.name += '_' + control_modes_dict[interaction_mode]
            int_marker.description = '3D Control'
            int_marker.description += '\n' + \
                control_modes_dict[interaction_mode]

        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 1.0
        control.orientation.y = 0.0
        control.orientation.z = 0.0
        self.normalize_quaternion(control.orientation)
        control.name = 'move_x'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 1.0
        control.orientation.y = 0.0
        control.orientation.z = 0.0
        self.normalize_quaternion(control.orientation)
        control.name = 'rotate_x'
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 1.0
        control.orientation.z = 0.0
        self.normalize_quaternion(control.orientation)
        control.name = 'move_y'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 1.0
        control.orientation.z = 0.0
        self.normalize_quaternion(control.orientation)
        control.name = 'rotate_y'
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 0.0
        control.orientation.z = 1.0
        self.normalize_quaternion(control.orientation)
        control.name = 'move_z'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 0.0
        control.orientation.z = 1.0
        self.normalize_quaternion(control.orientation)
        control.name = 'rotate_z'
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        self.server.insert(
            int_marker, feedback_callback=self.process_arrow_feedback)

        self.get_logger().info(f'Inserted marker: {int_marker.name}')


def main(args=None):
    rclpy.init(args=None)
    node = PoseMarker()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main(sys.argv)
