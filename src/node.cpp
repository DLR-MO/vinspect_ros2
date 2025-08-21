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

using namespace std::chrono_literals;
typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image>
  approx_policy;
/**
 * Converts a string array to a list of sensor type enums.
 *  @param string_array:
 * @return:
 */
std::vector<vinspect::SensorType> stringArrayToEnum(
  const std::vector<std::string> & string_array)
{
  std::vector<vinspect::SensorType> enums;
  for (const auto & s : string_array) {
    if (s == "SPARSE") {
      enums.push_back(vinspect::SensorType::SPARSE);
    } else if (s == "RGB") {
      enums.push_back(vinspect::SensorType::RGB);
    } else if (s == "DEPTH") {
      enums.push_back(vinspect::SensorType::DEPTH);
    } else if (s == "RGBD") {
      enums.push_back(vinspect::SensorType::RGBD);
    } else {
      throw std::runtime_error("Unknown sensor type: " + s);
    }
  }
  return enums;
}

double mean(std::vector<double> const & v)
{
  if (v.empty()) {
    return 0;
  }
  auto const count = static_cast<double>(v.size());
  return std::reduce(v.begin(), v.end()) / count;
}

bool compLess(double a, double b) {return a < b;}

class VinspectNode : public rclcpp::Node
{
public:
  VinspectNode()
  : Node("vinspect_node")
  {
    displayed_value_name_ = declare_parameter<std::string>("value_to_display", "");
    round_to_decimals_ = declare_parameter<int>("round_to_decimals", -1);
    save_path_ = declare_parameter<std::string>("save_path", "");
    frame_id_ = declare_parameter<std::string>("frame_id", "world");
    record_joints_ = declare_parameter<bool>("record_joints", false);
    current_ref_mesh_ = declare_parameter<std::string>("ref_mesh_path", "");
    std::vector<double> dense_senor_resolution = declare_parameter<std::vector<double>>(
      "dense_senor_resolution", {1.0, 1.0});
    std::vector<double> inspection_space_3d_min = declare_parameter<std::vector<double>>(
      "inspection_space_3d_min", {-1.0, -1.0, -1.0});
    std::vector<double> inspection_space_3d_max = declare_parameter<std::vector<double>>(
      "inspection_space_3d_max", {1.0, 1.0, 1.0});
    std::vector<double> inspection_space_6d_min = declare_parameter<std::vector<double>>(
      "inspection_space_6d_min", {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0});
    std::vector<double> inspection_space_6d_max = declare_parameter<std::vector<double>>(
      "inspection_space_6d_max", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    if (
      inspection_space_3d_min[0] < inspection_space_3d_max[0] &&
      inspection_space_3d_min[1] < inspection_space_3d_max[1] &&
      inspection_space_3d_min[2] < inspection_space_3d_max[2])
    {
      inspection_space_3d_min_ = {
        inspection_space_3d_min[0], inspection_space_3d_min[1], inspection_space_3d_min[2]};
      inspection_space_3d_max_ = {
        inspection_space_3d_max[0], inspection_space_3d_max[1], inspection_space_3d_max[2]};
    } else {
      RCLCPP_ERROR(get_logger(), "The 3D inspection space min and max values are not valid.");
      exit(1);
    }
    if (
      inspection_space_6d_min[0] < inspection_space_6d_max[0] &&
      inspection_space_6d_min[1] < inspection_space_6d_max[1] &&
      inspection_space_6d_min[2] < inspection_space_6d_max[2] &&
      inspection_space_6d_min[3] < inspection_space_6d_max[3] &&
      inspection_space_6d_min[4] < inspection_space_6d_max[4] &&
      inspection_space_6d_min[5] < inspection_space_6d_max[5])
    {
      inspection_space_6d_min_ = {
        inspection_space_6d_min[0], inspection_space_6d_min[1], inspection_space_6d_min[2],
        inspection_space_6d_min[3], inspection_space_6d_min[4], inspection_space_6d_min[5]};
      inspection_space_6d_max_ = {
        inspection_space_6d_max[0], inspection_space_6d_max[1], inspection_space_6d_max[2],
        inspection_space_6d_max[3], inspection_space_6d_max[4], inspection_space_6d_max[5]};
    } else {
      RCLCPP_ERROR(get_logger(), "The 6D inspection space min and max values are not valid.");
      exit(1);
    }

    std::vector<double> sparse_min_color_values = declare_parameter<std::vector<double>>(
      "sparse_min_color_values", std::vector<double>(0));
    std::vector<double> sparse_max_color_values = declare_parameter<std::vector<double>>(
      "sparse_max_color_values", std::vector<double>(0));

    std::string sparse_topic = declare_parameter<std::string>("sparse_topic", "sparse");
    std::string joint_topic = declare_parameter<std::string>("joint_topic", "joint_states");
    std::vector<std::string> sensor_type_strings = declare_parameter<std::vector<std::string>>(
      "sensor_types", {"SPARSE"});
    std::vector<vinspect::SensorType> sensor_types = stringArrayToEnum(sensor_type_strings);
    std::vector<std::string> sensor_data_type_names = declare_parameter<std::vector<std::string>>(
      "sensor_data_type_names", {"", ""});
    std::vector<std::string> sensor_data_type_units = declare_parameter<std::vector<std::string>>(
      "sensor_data_type_units", {"", ""});
    std::string load_path = declare_parameter<std::string>("load_path", "");

    std::vector<std::string> rgbd_color_topics = declare_parameter<std::vector<std::string>>(
      "rgbd_color_topics", {""});
    std::vector<std::string> rgbd_depth_topics = declare_parameter<std::vector<std::string>>(
      "rgbd_depth_topics", {""});
    std::vector<std::string> rgbd_info_topics = declare_parameter<std::vector<std::string>>(
      "rgbd_info_topics", {""});

    // Get joint names from robot_state_publisher
    if (record_joints_) {
      // todo Maybe get from robot_state_publisher? Normally easier to just get from topic
      // todo Let this listen for a few seconds, if no messages come print a warning
      joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        joint_topic, 10, std::bind(&VinspectNode::jointCb, this, std::placeholders::_1));
    }

    if (current_ref_mesh_.length() > 0) {
      std::string path = current_ref_mesh_;
      if (current_ref_mesh_.substr(0, 10) == "package://") {
        // This is a path based on a package in the format
        // package://package_name/path_inside_package
        std::string sub_path = current_ref_mesh_.substr(10);
        size_t end_index = sub_path.find('/');
        std::string package_name = sub_path.substr(0, end_index);
        std::string package_path = ament_index_cpp::get_package_share_directory(package_name);
        path = package_path + "/" + sub_path.substr(end_index + 1);
      }
      if (!open3d::io::ReadTriangleMesh(path, mesh_)) {
        RCLCPP_ERROR(this->get_logger(), "Error reading reference mesh.");
      }
    } else {
      RCLCPP_INFO(this->get_logger(), "No reference mesh path specified.");
    }

    if (load_path.length() > 0) {
      inspection_ = vinspect::load(load_path);
      // if we loaded data, we should not directly record data
      paused_ = true;
    } else if (std::filesystem::exists(save_path_)) {
      RCLCPP_WARN(
        this->get_logger(), "Loading already existing inspection at %s", save_path_.c_str());
      // todo we would need to check if any header information changed
      inspection_ = vinspect::load(save_path_);
    } else {
      std::vector<std::string> joint_names = {};                // todo
      inspection_ = vinspect::Inspection(
        sensor_types, sensor_data_type_names, sensor_data_type_units, joint_names, mesh_,
        std::make_tuple(
          dense_senor_resolution[0],
          dense_senor_resolution[1]),
        save_path_, inspection_space_3d_min_, inspection_space_3d_max_, inspection_space_6d_min_,
        inspection_space_6d_max_, sparse_min_color_values,
        sparse_max_color_values);
    }
    inspection_.startSaving();

    sparse_mesh_ = std::make_shared<vinspect::SparseMesh>(inspection_);

    // Don't use default callback group to allow parallel execution of multiple parts
    rclcpp::QoS latching_qos = rclcpp::QoS(1).transient_local();
    sparse_mesh_pub_ =
      this->create_publisher<visualization_msgs::msg::Marker>("mesh", latching_qos);
    dense_mesh_pub_ =
      this->create_publisher<visualization_msgs::msg::Marker>("dense_mesh", latching_qos);
    ref_marker_pub_ =
      this->create_publisher<visualization_msgs::msg::Marker>("ref_mesh", latching_qos);
    dense_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("dense_image", latching_qos);
    multi_dense_poses_pub_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>("visualization_marker_array",
      latching_qos);

    old_object_ = "Null";
    old_transparency_ = -0.1;
    last_mesh_number_sparse_ = -1;
    dot_size_ = 0.5 * 0.001;
    mean_min_max_ = vinspect_msgs::msg::Settings::MEAN;
    selection_sphere_radius_ = 0.02;

    mesh_marker_msg_.header.frame_id = frame_id_;
    mesh_marker_msg_.type = mesh_marker_msg_.TRIANGLE_LIST;
    mesh_marker_msg_.id = 1;
    mesh_marker_msg_.ns = "mesh";
    mesh_marker_msg_.lifetime = rclcpp::Duration::from_seconds(0.0);
    mesh_marker_msg_.frame_locked = true;
    mesh_marker_msg_.action = mesh_marker_msg_.MODIFY;
    mesh_marker_msg_.scale.x = 1.0;
    mesh_marker_msg_.scale.y = 1.0;
    mesh_marker_msg_.scale.z = 1.0;
    mesh_marker_msg_.pose.position.x = 0.0;
    mesh_marker_msg_.pose.position.y = 0.0;
    mesh_marker_msg_.pose.position.z = 0.0;
    mesh_marker_msg_.pose.orientation.w = 1.0;
    mesh_marker_msg_.pose.orientation.x = 0.0;
    mesh_marker_msg_.pose.orientation.y = 0.0;
    mesh_marker_msg_.pose.orientation.z = 0.0;
    // todo why does this data not show up? although we are transient local
    pubRefMesh(0.5);
    // if we loaded some data we might show it directly
    showCurrentData();

    status_pub_ = this->create_publisher<vinspect_msgs::msg::Status>("vinspect/status", 1);

    // Each subscriber has its own callback group. So different callbacks can be executed in
    // parallel, but not multiple times the same callback.
    rclcpp::SubscriptionOptions options1;
    options1.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    vis_params_sub_ = this->create_subscription<vinspect_msgs::msg::Settings>(
      "vinspect/settings", latching_qos,
      std::bind(&VinspectNode::settingsCb, this, std::placeholders::_1), options1);

    rclcpp::SubscriptionOptions options2;
    options2.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    // use keep all quality of service setting to avoid loosing data
    rclcpp::QoS keep_all_reliable_qos = rclcpp::QoS(1).keep_all().reliable();
    if (
      std::find(sensor_types.begin(), sensor_types.end(), vinspect::SensorType::SPARSE) !=
      sensor_types.end())
    {
      sparse_sub_ = this->create_subscription<vinspect_msgs::msg::Sparse>(
        sparse_topic, keep_all_reliable_qos,
        std::bind(&VinspectNode::sparseCb, this, std::placeholders::_1), options2);
    }

    rclcpp::SubscriptionOptions options3;
    options3.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    dense_req_sub = this->create_subscription<std_msgs::msg::String>(
      "vinspect/dense_data_req", latching_qos,
      std::bind(&VinspectNode::denseDataReq, this, std::placeholders::_1), options3);

    rclcpp::SubscriptionOptions options4;
    options4.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    multi_dense_req_sub = this->create_subscription<std_msgs::msg::Int32>(
      "vinspect/multi_dense_data_req", latching_qos,
      std::bind(&VinspectNode::multiDenseDataReq, this, std::placeholders::_1), options4);

    // todo should be true at the beginning and started with service call
    dense_pause_ = true;
    if (rgbd_color_topics.size() > 0 && rgbd_color_topics[0] != "") {
      // todo could be parameter, is currently updated by service
      depth_scale_ = 1000.0;
      depth_trunc_ = 3.0;
      tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
      start_reconstruction_service_ = this->create_service<vinspect_msgs::srv::StartReconstruction>(
        "/vinspect/start_reconstruction",
        std::bind(
          &VinspectNode::startReconstruction, this, std::placeholders::_1, std::placeholders::_2));
      stop_reconstruction_service_ = this->create_service<std_srvs::srv::Empty>(
        "/vinspect/stop_reconstruction",
        std::bind(
          &VinspectNode::stopReconstruction, this, std::placeholders::_1, std::placeholders::_2));
      // create subscriptions for any number of rgbd cameras
      if (
        rgbd_color_topics.size() !=
        rgbd_depth_topics.size())    // todo or != rgbd_info_topics.size()
      {
        RCLCPP_FATAL(this->get_logger(), "Number of RGBD color and depth topics do not match.");
        exit(1);
      }
      for (uint64_t i = 0; i < rgbd_color_topics.size(); i++) {
        // todo maybe we should use  image_transport::SubscriberFilter for more performance?
        color_subs_.push_back(
          std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>());
        depth_subs_.push_back(
          std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>());
        // todo maybe we need a common mutial exclusive callback group for these, as they all
        // acces the TSDF. or non exclusive groups?
        // todo specify qos and options as further arguments
        color_subs_[i]->subscribe(this, rgbd_color_topics[i].c_str());
        depth_subs_[i]->subscribe(this, rgbd_depth_topics[i].c_str());
        std::shared_ptr<message_filters::Synchronizer<approx_policy>> rgbd_sync =
          std::make_shared<message_filters::Synchronizer<approx_policy>>(
          approx_policy(100), *color_subs_[i].get(), *depth_subs_[i].get());
        rgbd_sync->getPolicy()->setMaxIntervalDuration(rclcpp::Duration::from_seconds(1.0 / 30.0));
        rgbd_sync->registerCallback(&VinspectNode::cameraCb, this);
        rgbd_syncs_.push_back(rgbd_sync);

        rgbd_info_subs_.push_back(
          this->create_subscription<sensor_msgs::msg::CameraInfo>(
            rgbd_info_topics[i], 10,
            std::bind(&VinspectNode::cameraInfoCb, this, std::placeholders::_1)));
      }
    }

    // handle interactive marker feedback
    display_data_pub_ = this->create_publisher<vinspect_msgs::msg::AreaData>("display_info", 2);
    rclcpp::SubscriptionOptions options5;
    options5.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    selection_marker_sub_ =
      this->create_subscription<visualization_msgs::msg::InteractiveMarkerFeedback>(
      "selection_marker/feedback", 10,
      std::bind(&VinspectNode::sparseInteractiveMarkerCb, this, std::placeholders::_1), options5);

    rclcpp::SubscriptionOptions options6;
    options6.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    selection_marker_sub_ =
      this->create_subscription<visualization_msgs::msg::InteractiveMarkerFeedback>(
      "pose_marker/feedback", 10,
      std::bind(&VinspectNode::denseInteractiveMarkerCb, this, std::placeholders::_1), options6);

    RCLCPP_INFO(this->get_logger(), "Inspection initialized. You can now start recording data.");
  }

  /**
   * Iterates over triangles and vertices of given mesh to add the color to the marker object.
   */
  void showCurrentData()
  {
    // only show this if the inspection has sparse data
    if (inspection_.getSparseUsage()) {
      mtx_.lock();
      // update if settings changed or new measurements are available
      // only compute anything if there is someone listening for the mesh
      if (
        (settings_changed_ || (last_mesh_number_sparse_ != inspection_.getSparseDataCount() &&
        inspection_.getSparseDataCount() != 0)) &&
        sparse_mesh_pub_->get_subscription_count() > 0)
      {
        settings_changed_ = false;
        const open3d::geometry::TriangleMesh mesh =
          sparse_mesh_->createMesh(dot_size_, use_custom_color_, mean_min_max_);
        last_mesh_number_sparse_ = inspection_.getSparseDataCount();
        mtx_.unlock();

        mesh_marker_msg_.points.clear();
        mesh_marker_msg_.colors.clear();
        for (uint64_t i = 0; i < mesh.triangles_.size(); i++) {
          for (int j = 0; j < 3; j++) {
            int idx = mesh.triangles_[i][j];

            geometry_msgs::msg::Point curr_point;
            curr_point.x = mesh.vertices_[idx].x();
            curr_point.y = mesh.vertices_[idx].y();
            curr_point.z = mesh.vertices_[idx].z();

            std_msgs::msg::ColorRGBA curr_point_color;
            curr_point_color.r = mesh.vertex_colors_[idx].x();
            curr_point_color.g = mesh.vertex_colors_[idx].y();
            curr_point_color.b = mesh.vertex_colors_[idx].z();
            curr_point_color.a = 1.0;

            mesh_marker_msg_.points.push_back(curr_point);
            mesh_marker_msg_.colors.push_back(curr_point_color);
          }
        }
        sparse_mesh_pub_->publish(mesh_marker_msg_);
      } else {
        mtx_.unlock();
      }
    }
  }

  void showStatus()
  {
    status_msg_.recorded_values = inspection_.getSparseDataCount();
    if (paused_) {
      status_msg_.status = "paused";
    } else {
      status_msg_.status = "running";
    }
    status_msg_.integrated_images = inspection_.getIntegratedImagesCount();
    if (dense_pause_) {
      status_msg_.dense_status = "paused";
    } else {
      status_msg_.dense_status = "running";
    }
    // todo maybe we should also send a status on shutdown
    status_pub_->publish(status_msg_);
  }

  void showTSDF()
  {
    // only compute anything if there is someone listening for the mesh
    if (dense_mesh_pub_->get_subscription_count() > 0 &&
      inspection_.getIntegratedImagesCount() > 0)
    {
      std::shared_ptr<open3d::geometry::TriangleMesh> mesh =
        inspection_.extractDenseReconstruction();
      visualization_msgs::msg::Marker mesh_msg = visualization_msgs::msg::Marker();
      mesh_msg.header.stamp = this->get_clock()->now();
      mesh_msg.header.frame_id = frame_id_;
      mesh_msg.type = mesh_msg.TRIANGLE_LIST;
      // todo maybe we need to remove the old one first
      mesh_msg.action = mesh_msg.ADD;
      mesh_msg.id = 1;
      mesh_msg.scale.x = 1.0;
      mesh_msg.scale.y = 1.0;
      mesh_msg.scale.z = 1.0;
      mesh_msg.pose.orientation.w = 1.0;
      for (Eigen::Vector3i triangle : mesh->triangles_) {
        for (int vertex_index : triangle) {
          geometry_msgs::msg::Point curr_point = geometry_msgs::msg::Point();
          curr_point.x = mesh->vertices_[vertex_index][0];
          curr_point.y = mesh->vertices_[vertex_index][1];
          curr_point.z = mesh->vertices_[vertex_index][2];
          std_msgs::msg::ColorRGBA curr_point_color = std_msgs::msg::ColorRGBA();
          curr_point_color.r = mesh->vertex_colors_[vertex_index][0];
          curr_point_color.g = mesh->vertex_colors_[vertex_index][1];
          curr_point_color.b = mesh->vertex_colors_[vertex_index][2];
          curr_point_color.a = 1.0;  // todo make this configurable like with sparse
          mesh_msg.points.push_back(curr_point);
          mesh_msg.colors.push_back(curr_point_color);
        }
      }
      dense_mesh_pub_->publish(mesh_msg);
    }
  }

  void finish()
  {
    mtx_.lock();
    inspection_.finish();
    mtx_.unlock();
  }

private:
  /**
   * Callback for joint topic. Adds incomung data to inspection.
   * @param msg JointState ROS message
   * @return
   */
  void jointCb(sensor_msgs::msg::JointState msg)
  {
    throw std::logic_error("Function not yet implemented");  // todo
  }

  /**
   * Callback for sparse topic. Adds incomung data to inspection.
   * @param msg Sparse ROS message
   * @return
   */
  void sparseCb(vinspect_msgs::msg::Sparse msg)
  {
    if (!paused_) {
      if (msg.header.frame_id != frame_id_) {
        RCLCPP_WARN_STREAM(
          this->get_logger(), "Frame id mismatch: " << msg.header.frame_id << " != " << frame_id_
                                                    << ". Ignoring message");
      }
      uint64_t timestamp = rclcpp::Time(msg.header.stamp).nanoseconds();
      double time_in_seconds = timestamp / 1000000000.0;
      int sensor_id = msg.sensor_id;
      std::array<double, 3> position = {
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z};
      std::array<double, 4> orientation = {
        msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y,
        msg.pose.orientation.z};
      std::vector<double> values = {};
      for (double v : msg.data) {
        values.push_back(v);
      }
      Eigen::Vector3d color =
        Eigen::Vector3d(msg.custom_color.r, msg.custom_color.g, msg.custom_color.b);
      if (msg.custom_color.a != 1.0) {
        RCLCPP_WARN(
          this->get_logger(),
          "Alpha channel is not supported. Please set the alpha channel to 1.0");
      }
      mtx_.lock();
      inspection_.addSparseMeasurement(
        time_in_seconds, sensor_id, position, orientation, values, color);
      mtx_.unlock();
    }
  }

  void sparseInteractiveMarkerCb(visualization_msgs::msg::InteractiveMarkerFeedback feedback)
  {
    if (feedback.event_type == visualization_msgs::msg::InteractiveMarkerFeedback::MOUSE_UP) {
      std::array<double, 3> point_coords = {
        feedback.pose.position.x, feedback.pose.position.y, feedback.pose.position.z};
      mtx_.lock();
      if (inspection_.getSparseDataCount() == 0) {
        mtx_.unlock();
        display_data_msg_.in_area = "No measurement";
        display_data_msg_.next_neighbor = "No measurement";
        display_data_msg_.mean = "No measurement";
        display_data_msg_.min = "No measurement";
        display_data_msg_.max = "No measurement";
        display_data_pub_->publish(display_data_msg_);
      } else {
        std::vector<uint64_t> points_in_area =
          inspection_.getSparseMeasurementsInRadius(point_coords, selection_sphere_radius_);
        uint64_t closests_point = inspection_.getClosestSparseMeasurement(point_coords);
        std::vector<double> values_in_area =
          inspection_.getValuesForIds(displayed_value_name_, points_in_area);
        std::vector<std::string> units = inspection_.getSparseUnits();
        mtx_.unlock();

        double mean_value = 0;
        double min_value = 0;
        double max_value = 0;
        std::string unit = units[0];  // todo the index 0 should be chosen dynamically
        if (values_in_area.size() > 0) {
          mean_value = mean(values_in_area);
          auto minmax = std::minmax_element(values_in_area.begin(), values_in_area.end(), compLess);
          min_value = *minmax.first;
          max_value = *minmax.second;
        }
        display_data_msg_.in_area = std::to_string(points_in_area.size());
        display_data_msg_.next_neighbor = std::to_string(closests_point);
        // todo the rounding should rather be done in the visualization
        display_data_msg_.mean = std::to_string(roundValue(mean_value)) + unit;
        display_data_msg_.min = std::to_string(roundValue(min_value)) + unit;
        display_data_msg_.max = std::to_string(roundValue(max_value)) + unit;
        display_data_pub_->publish(display_data_msg_);
      }
    }
  }

  /**
   * Orders the image from a std::vector<std::vector<std::array<u_int8_t,3>>>
   * To sensor_msgs::msg::Image type.
   * @param image in as vector
   * @return image as sensor_msgs::msg::Image
   */
  sensor_msgs::msg::Image vectorToImageMsg(
    const std::vector<std::vector<std::array<u_int8_t,
    3>>> & image)
  {
    sensor_msgs::msg::Image msg;
    msg.height = image.size();
    msg.width = image[1].size();

    msg.encoding = "rgb8";

    msg.step = msg.width * 3;

    msg.data.resize(msg.height * msg.step);
    for (size_t i = 0; i < msg.height; i++) {
      for (size_t j = 0; j < msg.width; j++) {
        size_t index = (i * msg.width + j) * 3;
        msg.data[index + 0] = image[i][j][0];
        msg.data[index + 1] = image[i][j][1];
        msg.data[index + 2] = image[i][j][2];
      }
    }
    return msg;
  }
  /**
   * Callback for the denseInteractiveMarker. Used to set the current pose.
   * @param feedback InteractiveMarkerFeedback message
   * @return None
   */
  void denseInteractiveMarkerCb(visualization_msgs::msg::InteractiveMarkerFeedback feedback)
  {
    if (feedback.event_type == visualization_msgs::msg::InteractiveMarkerFeedback::MOUSE_UP) {
      dense_interactive_marker_pose_ = {
        feedback.pose.position.x, feedback.pose.position.y, feedback.pose.position.z,
        feedback.pose.orientation.x, feedback.pose.orientation.y, feedback.pose.orientation.z,
        feedback.pose.orientation.w};
    }
  }

  /**
   * Publishes a reference mesh object for dense data inspection.
   * Deletes the old dense reference mesh first.
   *
   * @param pose Pose of the marker with orientation as a quaternion
   * @return None
   */
  void pubRefMeshDense(std::array<double, 7> pose)
  {
    visualization_msgs::msg::Marker marker = visualization_msgs::msg::Marker();
    marker.header.frame_id = "world";
    marker.id = 1;
    marker.ns = "dense_ref_mesh";
    marker.action = marker.DELETE;
    // Delete the old marker first, out of some reason, MODIFY does not work
    ref_marker_pub_->publish(marker);
    marker.action = marker.ADD;
    marker.frame_locked = true;
    // marker.mesh_use_embedded_materials = true;
    marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
    marker.mesh_resource = "package://vinspect_ros2/data/camera_1.stl";
    marker.scale.x = 0.002;
    marker.scale.y = 0.002;
    marker.scale.z = 0.002;
    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
    marker.color.a = 1.0;
    marker.pose.position.x = pose[0];
    marker.pose.position.y = pose[1];
    marker.pose.position.z = pose[2];

    marker.pose.orientation.x = pose[3];
    marker.pose.orientation.y = pose[4];
    marker.pose.orientation.z = pose[5];
    marker.pose.orientation.w = pose[6];

    ref_marker_pub_->publish(marker);
  }

  /**
   * Callback for message on "vinspect/dense_data_req"
   * @param None Currently not used
   * @return None
   */
  // Note: If it stays like this, it could also be a service/client call. Thought we have more
  // options in the future if this message is used as a settings string or something similar later.
  void denseDataReq(std_msgs::msg::String)
  {
    mtx_.lock();
    if (inspection_.getDenseDataCount() == 0) {
      mtx_.unlock();
      RCLCPP_INFO(this->get_logger(), "No dense data available");
    } else {
      RCLCPP_INFO(this->get_logger(), "Asking vinspect for images");
      int id_to_get = inspection_.getClosestDenseMeasurement(dense_interactive_marker_pose_);
      auto image = inspection_.getImageFromId(id_to_get);
      auto pose = inspection_.getDensePoseFromId(id_to_get);
      pubRefMeshDense(inspection_.eulerToQuatPose(pose));
      auto msg = vectorToImageMsg(image);
      dense_image_pub_->publish(msg);
      RCLCPP_INFO(this->get_logger(), "Published image");
      mtx_.unlock();
    }
  }

  double roundValue(double value)
  {
    if (round_to_decimals_ < 0) {
      // no rounding
      return value;
    } else {
      // round to number of decimals after the decimal point
      return round(value * pow(10, round_to_decimals_)) / pow(10, round_to_decimals_);
    }
  }

  void multiDenseDataReq(std_msgs::msg::Int32 msg)
  {
    std::vector<std::array<double, 6>> poses = inspection_.getMultiDensePoses(msg.data);

    visualization_msgs::msg::MarkerArray markerArr = visualization_msgs::msg::MarkerArray();
    for (size_t i = 0; i < poses.size(); i++) {
      std::array<double, 7> quat_pose = inspection_.eulerToQuatPose(poses[i]);
      visualization_msgs::msg::Marker marker = visualization_msgs::msg::Marker();
      marker.header.frame_id = "world";
      marker.type = visualization_msgs::msg::Marker::ARROW;
      marker.id = i;
      marker.ns = "allDensePoses";
      marker.action = marker.ADD;
      marker.frame_locked = true;

      marker.pose.position.x = quat_pose[0];
      marker.pose.position.y = quat_pose[1];
      marker.pose.position.z = quat_pose[2];
      marker.pose.orientation.x = quat_pose[3];
      marker.pose.orientation.y = quat_pose[4];
      marker.pose.orientation.z = quat_pose[5];
      marker.pose.orientation.w = quat_pose[6];
      marker.scale.x = 0.05;
      marker.scale.y = 0.005;
      marker.scale.z = 0.005;
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;
      markerArr.markers.push_back(marker);
    }
    multi_dense_poses_pub_->publish(markerArr);
    RCLCPP_INFO(this->get_logger(), "Showing: requested available poses.");
  }

  /**
   * Publishes a reference mesh object.
   * Deletes the old reference mesh first.
   *
   * @param transparency How transparent the mesh should be
   * @return None
   */
  void pubRefMesh(float transparency)
  {
    visualization_msgs::msg::Marker marker = visualization_msgs::msg::Marker();
    marker.header.frame_id = frame_id_;
    marker.type = 10;
    marker.id = 0;
    marker.ns = "ref_mesh";
    marker.action = marker.DELETE;
    // Delete the old marker first, out of some reason, MODIFY does not work
    ref_marker_pub_->publish(marker);

    marker.action = marker.ADD;
    marker.frame_locked = true;
    marker.mesh_resource = current_ref_mesh_;
    marker.mesh_use_embedded_materials = true;
    marker.scale.x = 0.001;
    marker.scale.y = 0.001;
    marker.scale.z = 0.001;
    marker.color.a = transparency;
    ref_marker_pub_->publish(marker);
    old_transparency_ = transparency;
  }

  /**
   * Is given to mesh_subscriber and called on an incoming message.
   * Publishes a reference mesh object if the request in the msg is different than before.
   *
   * @param msg Incoming message to handle.
   * @return None
   */
  void settingsCb(vinspect_msgs::msg::Settings msg)
  {
    if (old_transparency_ != msg.transparency) {
      pubRefMesh(msg.transparency);
    }

    if (msg.dot_size > 0) {
      dot_size_ = msg.dot_size;
    }
    if (msg.sphere_radius > 0) {
      selection_sphere_radius_ = msg.sphere_radius;
    }
    mean_min_max_ = msg.mean_min_max;
    use_custom_color_ = msg.custom_color;
    if (msg.pause) {
      paused_ = true;
    }
    if (msg.resume) {
      paused_ = false;
    }
    if (msg.clear) {
      paused_ = true;
      mtx_.lock();
      inspection_.clear();
      sparse_mesh_->resetMesh();
      mesh_marker_msg_.points.clear();
      mesh_marker_msg_.colors.clear();
      sparse_mesh_pub_->publish(mesh_marker_msg_);
      mtx_.unlock();
    }
    settings_changed_ = true;
  }

  void cameraInfoCb(sensor_msgs::msg::CameraInfo msg)
  {
    // todo maybe we should provide them at the construction of the inspection object
    //  todo need to differentiate between different cameras
    open3d::camera::PinholeCameraIntrinsic intrinsic = open3d::camera::PinholeCameraIntrinsic(
      msg.width, msg.height, msg.k[0], msg.k[4], msg.k[2], msg.k[5]);
    inspection_.setIntrinsic(intrinsic, 0);
  }

  void cameraCb(
    const sensor_msgs::msg::Image::ConstSharedPtr & color_image_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_image_msg)
  {
    if (!dense_pause_) {
      open3d::geometry::Image o3d_color_img;
      open3d::geometry::Image o3d_depth_img;
      try {
        // todo check if this could be done with better performance
        //  convert ROS image message to opencv
        cv_bridge::CvImageConstPtr cv2_color_img =
          cv_bridge::toCvShare(color_image_msg, color_image_msg->encoding);
        cv_bridge::CvImageConstPtr cv2_depth_img =
          cv_bridge::toCvShare(depth_image_msg, std::string("16UC1"));
        // convert opencv image to open3d image
        // Allocate data buffer
        o3d_color_img.Prepare(color_image_msg->width, color_image_msg->height, 3, 1);
        o3d_depth_img.Prepare(depth_image_msg->width, depth_image_msg->height, 1, 2);
        // copy data from opencv image to open3d image
        memcpy(o3d_depth_img.data_.data(), cv2_depth_img->image.data, o3d_depth_img.data_.size());
        memcpy(o3d_color_img.data_.data(), cv2_color_img->image.data, o3d_color_img.data_.size());
      } catch (cv_bridge::Exception & e) {
        RCLCPP_ERROR(this->get_logger(), "Error converting image from ROS to CV");
        return;
      }

      geometry_msgs::msg::TransformStamped transformed_pose_optical;
      geometry_msgs::msg::TransformStamped transformed_pose_world;
      try {
        transformed_pose_optical = tf_buffer_->lookupTransform(
          color_image_msg->header.frame_id, frame_id_, color_image_msg->header.stamp);

        /* Note: It is expected that an equivalent non-optical frame exists
        to the optical frame in which the image is published. */
        std::string non_optical_frame = removeWordFromString(
          color_image_msg->header.frame_id,
          "_optical");
        assert(!non_optical_frame.empty());
        transformed_pose_world = tf_buffer_->lookupTransform(
          frame_id_, non_optical_frame, color_image_msg->header.stamp);
      } catch (tf2::TransformException & e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get transform: %s", e.what());
        return;
      }

      Eigen::Matrix4d rgb_pose_tsdf = transformStampedToTransformMatix(transformed_pose_optical);
      Eigen::Matrix4d rgb_pose_world = transformStampedToTransformMatix(transformed_pose_world);

      std::shared_ptr<open3d::geometry::RGBDImage> rgbd =
        open3d::geometry::RGBDImage::CreateFromColorAndDepth(
        o3d_color_img, o3d_depth_img, depth_scale_, depth_trunc_, false);
      // open3d::visualization::DrawGeometries({rgbd});
      //  todo sensor id should not be hardcoded to 0
      inspection_.integrateImage(*rgbd.get(), 0, rgb_pose_tsdf, rgb_pose_world);
    }
  }

  std::string removeWordFromString(const std::string str, const std::string word)
  {
    std::string to_remove = word;
    std::string remove_from = str;
    std::string result;
    auto n = remove_from.find(to_remove);
    if (n != std::string::npos) {
      result = remove_from.erase(n, to_remove.length());
    }
    return result;
  }

  /**
   * Converts from transformStamped message to Eigen::Matrix4d transform matrix.
   * @param transformed_pose transformStamped message
   * @return transform matrix as Eigen::Matrix4d
   */
  Eigen::Matrix4d transformStampedToTransformMatix(
    const geometry_msgs::msg::TransformStamped transformed_pose)
  {
    tf2::Quaternion quat = tf2::Quaternion(
      transformed_pose.transform.rotation.x, transformed_pose.transform.rotation.y,
      transformed_pose.transform.rotation.z, transformed_pose.transform.rotation.w);
    tf2::Matrix3x3 mat = tf2::Matrix3x3(quat);

    Eigen::Matrix4d transform_m;

    // create Transformation matrix
    transform_m(0, 0) = mat[0][0];
    transform_m(0, 1) = mat[0][1];
    transform_m(0, 2) = mat[0][2];
    transform_m(1, 0) = mat[1][0];
    transform_m(1, 1) = mat[1][1];
    transform_m(1, 2) = mat[1][2];
    transform_m(2, 0) = mat[2][0];
    transform_m(2, 1) = mat[2][1];
    transform_m(2, 2) = mat[2][2];
    transform_m(0, 3) = transformed_pose.transform.translation.x;
    transform_m(1, 3) = transformed_pose.transform.translation.y;
    transform_m(2, 3) = transformed_pose.transform.translation.z;
    transform_m(3, 0) = 0;
    transform_m(3, 1) = 0;
    transform_m(3, 2) = 0;
    transform_m(3, 3) = 1;

    return transform_m;
  }

  void startReconstruction(
    const std::shared_ptr<vinspect_msgs::srv::StartReconstruction::Request> request,
    std::shared_ptr<vinspect_msgs::srv::StartReconstruction::Response> response)
  {
    if (!dense_pause_) {
      response->success = false;
      RCLCPP_ERROR(this->get_logger(), "Reconstruction is already running");
    } else {
      depth_scale_ = request->depth_scale;
      depth_trunc_ = request->depth_trunc;
      inspection_.reinitializeTSDF(request->voxel_length, request->sdf_trunc);
      dense_pause_ = false;
    }
  }
  // todo maybe add continue as an option for tsdf

  void stopReconstruction(
    const std::shared_ptr<std_srvs::srv::Empty::Request> request,
    std::shared_ptr<std_srvs::srv::Empty::Response> response)
  {
    dense_pause_ = true;
  }

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

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // init node
  auto node = std::make_shared<VinspectNode>();
  rclcpp::Duration timer_duration = rclcpp::Duration::from_seconds(0.1);
  // timer for mesh publishing with own callback group to ensure that it runs parallel to the
  // rest of the node, but not multiple times parallel to itself
  rclcpp::CallbackGroup::SharedPtr timer_data_callback_group =
    node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::TimerBase::SharedPtr timer_data = rclcpp::create_timer(
    node, node->get_clock(), timer_duration, [node]() -> void {node->showCurrentData();},
    timer_data_callback_group);
  rclcpp::CallbackGroup::SharedPtr timer_status_callback_group =
    node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::TimerBase::SharedPtr timer_status = rclcpp::create_timer(
    node, node->get_clock(), rclcpp::Duration::from_seconds(0.1),
    [node]() -> void {node->showStatus();}, timer_status_callback_group);
  rclcpp::CallbackGroup::SharedPtr timer_tsdf_callback_group =
    node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::TimerBase::SharedPtr timer_tsdf = rclcpp::create_timer(
    node, node->get_clock(), rclcpp::Duration::from_seconds(1.0),
    [node]() -> void {node->showTSDF();}, timer_tsdf_callback_group);
  rclcpp::experimental::executors::EventsExecutor exec =
    rclcpp::experimental::executors::EventsExecutor();
  exec.add_node(node);

  exec.spin();
  node->finish();
  rclcpp::shutdown();
}
