cmake_minimum_required(VERSION 3.5)
project(vinspect_ros2)

set(CMAKE_CXX_STANDARD 20)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Open3D REQUIRED)
find_package(vinspect REQUIRED)
find_package(TBB REQUIRED)

set(ROS2_DEPS
  ament_index_cpp
  cv_bridge
  message_filters
  rclcpp
  sensor_msgs
  std_msgs
  std_srvs
  tf2_ros
  vinspect_msgs
  visualization_msgs
  tf2_geometry_msgs
)

foreach(dep ${ROS2_DEPS})
  find_package(${dep} REQUIRED)
endforeach()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused)

set(SOURCES src/node.cpp)

add_executable(node src/node.cpp)
target_link_libraries(node PUBLIC TBB::tbb)
ament_target_dependencies(node PUBLIC ${ROS2_DEPS} Open3D vinspect)
target_include_directories(node PUBLIC vinspect)

install(TARGETS node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/dummy_publisher.py scripts/selection_marker.py scripts/pose_marker.py scripts/load_and_publish.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY demo/launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY demo/data DESTINATION share/${PROJECT_NAME})
install(DIRECTORY demo/config DESTINATION share/${PROJECT_NAME})
ament_export_dependencies(vinspect)
ament_package()
