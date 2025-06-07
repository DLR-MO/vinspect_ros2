// SPDX-FileCopyrightText: 2024 Marc Bestmann <marc.bestmann@dlr.de>
//
// SPDX-License-Identifier: MIT

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <open3d/Open3D.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_srvs/srv/empty.hpp>
#include <visualization_msgs/msg/interactive_marker_feedback.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "vinspect/sparse_mesh.hpp"
#include "vinspect/inspection.hpp"
#include "vinspect_msgs/msg/area_data.hpp"
#include "vinspect_msgs/msg/settings.hpp"
#include "vinspect_msgs/msg/sparse.hpp"
#include "vinspect_msgs/msg/status.hpp"

#include "vinspect_msgs/srv/start_reconstruction.hpp"

typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image>
  approx_policy;

class VinspectNode : public rclcpp::Node
{
public:
  VinspectNode(const rclcpp::NodeOptions & options);
  void finish();

private:
  void showCurrentData();
  void showStatus();
  void showTSDF();
  void jointCb(sensor_msgs::msg::JointState msg);
  void sparseCb(vinspect_msgs::msg::Sparse msg);
  void sparseInteractiveMarkerCb(
    visualization_msgs::msg::InteractiveMarkerFeedback feedback);
  sensor_msgs::msg::Image vectorToImageMsg(
    const std::vector<std::vector<std::array<u_int8_t,
    3>>> & image);
  void denseInteractiveMarkerCb(
    visualization_msgs::msg::InteractiveMarkerFeedback feedback);
  void pubRefMeshDense(std::array<double, 7> pose);
  void denseDataReq(std_msgs::msg::String);
  double roundValue(double value);
  void multiDenseDataReq(std_msgs::msg::Int32 msg);
  void pubRefMesh(float transparency);
  void settingsCb(vinspect_msgs::msg::Settings msg);
  void cameraInfoCb(sensor_msgs::msg::CameraInfo msg);
  void cameraCb(
    const sensor_msgs::msg::Image::ConstSharedPtr & color_image_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_image_msg);
  std::string removeWordFromString(const std::string str, const std::string word);
  Eigen::Matrix4d transformStampedToTransformMatix(
    const geometry_msgs::msg::TransformStamped transformed_pose);
  void startReconstruction(
    const std::shared_ptr<vinspect_msgs::srv::StartReconstruction::Request> request,
    std::shared_ptr<vinspect_msgs::srv::StartReconstruction::Response> response);
  void stopReconstruction(
    const std::shared_ptr<std_srvs::srv::Empty::Request> request,
    std::shared_ptr<std_srvs::srv::Empty::Response> response);

  vinspect::Inspection inspection_;
  std::shared_ptr<vinspect::SparseMesh> sparse_mesh_;
  std::string save_path_;
  std::string frame_id_;
  bool record_joints_;
  open3d::geometry::TriangleMesh mesh_;
  std::string current_ref_mesh_;
  std::array<double, 3> inspection_space_3d_min_;
  std::array<double, 3> inspection_space_3d_max_;
  std::array<double, 6> inspection_space_6d_min_;
  std::array<double, 6> inspection_space_6d_max_;
  int round_to_decimals_;

  std::string old_object_;
  double old_transparency_;
  uint64_t last_mesh_number_sparse_;
  double dot_size_;
  double selection_sphere_radius_;
  int mean_min_max_;
  bool use_custom_color_;
  std::string displayed_value_name_;
  bool paused_;
  bool dense_pause_;
  bool settings_changed_;
  std::mutex mtx_;
  visualization_msgs::msg::Marker mesh_marker_msg_;
  vinspect_msgs::msg::AreaData display_data_msg_;
  vinspect_msgs::msg::Status status_msg_;

  double depth_scale_;
  double depth_trunc_;

  std::array<double, 7> dense_interactive_marker_pose_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};

  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr ref_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr sparse_mesh_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr dense_mesh_pub_;
  rclcpp::Publisher<vinspect_msgs::msg::AreaData>::SharedPtr display_data_pub_;
  rclcpp::Publisher<vinspect_msgs::msg::Status>::SharedPtr status_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr dense_image_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr multi_dense_poses_pub_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
  rclcpp::Subscription<vinspect_msgs::msg::Sparse>::SharedPtr sparse_sub_;
  std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>>> color_subs_;
  std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>>> depth_subs_;
  std::vector<std::shared_ptr<message_filters::Synchronizer<approx_policy>>> rgbd_syncs_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> rgbd_info_subs_;
  rclcpp::Subscription<vinspect_msgs::msg::Settings>::SharedPtr vis_params_sub_;
  rclcpp::Subscription<visualization_msgs::msg::InteractiveMarkerFeedback>::SharedPtr
    selection_marker_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr dense_req_sub;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr multi_dense_req_sub;

  rclcpp::Service<vinspect_msgs::srv::StartReconstruction>::SharedPtr start_reconstruction_service_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr stop_reconstruction_service_;

};
