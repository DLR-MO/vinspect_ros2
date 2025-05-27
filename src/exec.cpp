// SPDX-FileCopyrightText: 2024 Marc Bestmann <marc.bestmann@dlr.de>
//
// SPDX-License-Identifier: MIT

#include <rclcpp/rclcpp.hpp>
#include "vinspect_ros2/node.hpp"

// custom executable to use the EventExecutor in a component container
// can be deleted when https://github.com/ros2/rclcpp/pull/2541 is merged
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // init node
  rclcpp::NodeOptions options;
  auto node = std::make_shared<VinspectNode>(options);
  rclcpp::experimental::executors::EventsExecutor exec =
    rclcpp::experimental::executors::EventsExecutor();
  exec.add_node(node);

  exec.spin();
  node->finish();
  rclcpp::shutdown();
}
