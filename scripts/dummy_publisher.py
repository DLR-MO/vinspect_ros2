#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2024 Marc Bestmann <marc.bestmann@dlr.de>
#
# SPDX-License-Identifier: MIT

import random

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from vinspect_msgs.msg import Sparse

if __name__ == '__main__':
    rclpy.init(args=None)
    node = Node('dummy_publisher')
    publisher = node.create_publisher(Sparse, 'sparse', 10)
    i = 0
    while rclpy.ok():
        msg = Sparse()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = random.uniform(-1.0, 1.0)
        msg.pose.position.y = random.uniform(-1.0, 1.0)
        msg.pose.position.z = random.uniform(0.001, 0.001)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        msg.data = [random.uniform(0, 1.0)]
        msg.custom_color.r = random.uniform(0.0, 1.0)
        msg.custom_color.g = random.uniform(0.0, 1.0)
        msg.custom_color.b = random.uniform(0.0, 1.0)
        msg.custom_color.a = 1.0
        i += 1
        publisher.publish(msg)
        node.get_clock().sleep_for(Duration(seconds=1))
    rclpy.shutdown()
