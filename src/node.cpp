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
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/empty.hpp>
#include <visualization_msgs/msg/interactive_marker_feedback.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "vinspect/sparse_mesh.hpp"
#include "vinspect/inspection.hpp"
#include "vinspect/sensors.hpp"
#include "vinspect/utils.hpp"
#include "vinspect_msgs/msg/area_data.hpp"
#include "vinspect_msgs/msg/settings.hpp"
#include "vinspect_msgs/msg/sparse.hpp"
#include "vinspect_msgs/msg/status.hpp"

#include "vinspect_msgs/srv/start_reconstruction.hpp"

#include <vinspect_ros2/vinspect_parameters.hpp> 

using namespace std::chrono_literals;
typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image>
  approx_policy;

double mean(std::vector<double> const & v)
{
  if (v.empty()) {
    return 0;
  }
  auto const count = static_cast<double>(v.size());
  return std::reduce(v.begin(), v.end()) / count;
}

class VinspectNode : public rclcpp::Node
{
  public:
  VinspectNode()
  : Node("vinspect_node"), 
    param_listener_{std::make_shared<vinspect::ParamListener>(get_node_parameters_interface())},
    params_{param_listener_->get_params()}
  {
    
    // Check if the workspace bounds make sense
    for (int i = 0; i < 3; ++i) {
      if (params_.inspection_space_3d.min[i] > params_.inspection_space_3d.max[i])
      {
        RCLCPP_ERROR(get_logger(), "The 3D inspection space min and max values are not valid.");
        exit(1);
      }
    }
    for (int i = 0; i < 6; ++i) {
      if (params_.inspection_space_6d.min[i] > params_.inspection_space_6d.max[i])
      {
        RCLCPP_ERROR(get_logger(), "The 6D inspection space min and max values are not valid.");
        exit(1);
      }
    }

    // Get joint names from robot_state_publisher
    if (params_.joints_topic.size() > 0) {
      // todo Maybe get from robot_state_publisher? Normally easier to just get from topic
      // todo Let this listen for a few seconds, if no messages come print a warning
      joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        params_.joints_topic, 10, std::bind(&VinspectNode::jointCb, this, std::placeholders::_1));
    }

    // Check if file exists
    if (std::filesystem::exists(params_.save_path)) {
      inspection_ = std::make_unique<vinspect::Inspection>(params_.save_path);
      // if we loaded data, we should not directly record data
      paused_ = true;
    } else {
      // Create sensor objects
      std::vector<vinspect::SparseValueInfo> sparse_value_infos;      
      for (std::size_t i = 0; i < params_.sparse.value_names.size(); ++i) {
        sparse_value_infos.push_back(
          vinspect::SparseValueInfo{
            .name = params_.sparse.value_names[i],
            .unit = params_.sparse.value_units[i]
          }
        );
      }

      std::vector<vinspect::DenseSensor> dense_sensors;
      for (const auto& name : params_.dense_sensor_names) {
        auto sensor_params = params_.dense_sensor_names_map.at(name);
        dense_sensors.emplace_back(
          std::stoi(name),
          sensor_params.width,
          sensor_params.height,
          sensor_params.depth_scale
        ); 
      }

      // Resolve mesh path
      std::string ref_mesh_path = params_.ref_mesh_path;
      if (ref_mesh_path.starts_with("package://")) {
        // Create stream and drop the prefix
        std::stringstream stream(ref_mesh_path.substr(10)); 
        // Extract the package name
        std::string package_name;
        std::getline(stream, package_name, '/');
        std::string path_in_package;
        std::getline(stream, path_in_package);
        // Combine the package path and the path in the package
        ref_mesh_path = ament_index_cpp::get_package_share_directory(package_name) + "/" + path_in_package;
      }

      inspection_ = std::make_unique<vinspect::Inspection>(
        sparse_value_infos,
        dense_sensors,
        ref_mesh_path,
        params_.save_path, 
        vinspect::vec2array<double, 3>(params_.inspection_space_3d.min), 
        vinspect::vec2array<double, 3>(params_.inspection_space_3d.max), 
        vinspect::vec2array<double, 6>(params_.inspection_space_6d.min),
        vinspect::vec2array<double, 6>(params_.inspection_space_6d.max)
      );
    }

    sparse_mesh_ = std::make_unique<vinspect::SparseMesh>(*inspection_);

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

    old_transparency_ = -0.1;
    last_mesh_number_sparse_ = -1;
    dot_size_ = 0.5 * 0.001;
    mean_min_max_ = vinspect_msgs::msg::Settings::MEAN;
    selection_sphere_radius_ = 0.02;

    mesh_marker_msg_.header.frame_id = params_.frame_id;
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

    // Create sparse subscriptions
    if (inspection_->getSparseUsage()) {
      sparse_sub_ = this->create_subscription<vinspect_msgs::msg::Sparse>(
        params_.sparse.topic,
        keep_all_reliable_qos,
        std::bind(&VinspectNode::sparseCb, this, std::placeholders::_1),
        options2);
    }

    rclcpp::SubscriptionOptions options3;
    options3.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    dense_req_sub = this->create_subscription<std_msgs::msg::Empty>(
      "vinspect/dense_data_req", latching_qos,
      std::bind(&VinspectNode::denseDataReq, this, std::placeholders::_1), options3);

    rclcpp::SubscriptionOptions options4;
    options4.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    multi_dense_req_sub = this->create_subscription<std_msgs::msg::Int32>(
      "vinspect/multi_dense_data_req", latching_qos,
      std::bind(&VinspectNode::multiDenseDataReq, this, std::placeholders::_1), options4);

    if (inspection_->getDenseUsage()) {
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
      for (const auto& name : params_.dense_sensor_names) {
        auto sensor_params = params_.dense_sensor_names_map.at(name);

        // todo maybe we should use  image_transport::SubscriberFilter for more performance?
        auto& color_sub = color_subs_.emplace_back(std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>());
        auto& depth_sub = depth_subs_.emplace_back(std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>());

        color_sub->subscribe(this, sensor_params.color_topic);
        depth_sub->subscribe(this, sensor_params.depth_topic);

        // todo maybe we need a common mutial exclusive callback group for these, as they all
        // access the TSDF. or non exclusive groups?  No, because the events executor is single threaded
        // todo specify qos and options as further arguments
        std::shared_ptr<message_filters::Synchronizer<approx_policy>> rgbd_sync =
          std::make_shared<message_filters::Synchronizer<approx_policy>>(
          approx_policy(100), *color_sub.get(), *depth_sub.get());
        rgbd_sync->getPolicy()->setMaxIntervalDuration(rclcpp::Duration::from_seconds(1.0 / 30.0));
        rgbd_sync->registerCallback(std::bind(&VinspectNode::cameraCb, this, std::placeholders::_1, std::placeholders::_2, name));
        rgbd_syncs_.push_back(rgbd_sync);

        rgbd_info_subs_.push_back(
          this->create_subscription<sensor_msgs::msg::CameraInfo>(
            sensor_params.camera_info_topic, 
            10, 
            [this, name](const sensor_msgs::msg::CameraInfo msg) {
              cameraInfoCb(msg, name);
            }));
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
    pose_marker_sub_ =
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
    if (inspection_->getSparseUsage()) {
      mtx_.lock();
      // update if settings changed or new measurements are available
      // only compute anything if there is someone listening for the mesh
      if (
        (settings_changed_ || (last_mesh_number_sparse_ != inspection_->getSparseDataCount() &&
        inspection_->getSparseDataCount() != 0)) &&
        sparse_mesh_pub_->get_subscription_count() > 0)
      {
        settings_changed_ = false;
        const open3d::geometry::TriangleMesh mesh =
          sparse_mesh_->createMesh(dot_size_, use_custom_color_, mean_min_max_);
        last_mesh_number_sparse_ = inspection_->getSparseDataCount();
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
    status_msg_.recorded_values = inspection_->getSparseDataCount();
    if (paused_) {
      status_msg_.status = "paused";
    } else {
      status_msg_.status = "running";
    }
    status_msg_.integrated_images = inspection_->getIntegratedImagesCount();
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
      inspection_->getIntegratedImagesCount() > 0)
    {
      std::shared_ptr<open3d::geometry::TriangleMesh> mesh =
        inspection_->extractDenseReconstruction();

      if(mesh->triangles_.size() == 0) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
          "Reconstructed mesh is empty. Check if your depth scale and your inspection space are correct.");
        return;
      }

      visualization_msgs::msg::Marker mesh_msg;
      mesh_msg.header.stamp = this->get_clock()->now();
      mesh_msg.header.frame_id = params_.frame_id;
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
   * Callback for sparse topic. Adds incoming data to inspection.
   * @param msg Sparse ROS message
   * @return
   */
  void sparseCb(const vinspect_msgs::msg::Sparse msg)
  {
    if (!paused_) {
      if (msg.header.frame_id != params_.frame_id) {
        RCLCPP_WARN_STREAM(
          this->get_logger(), "Frame id mismatch: " << msg.header.frame_id << " != " << params_.frame_id
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
      inspection_->addSparseMeasurement(
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
      if (inspection_->getSparseDataCount() == 0) {
        mtx_.unlock();
        display_data_msg_.in_area = "No measurement";
        display_data_msg_.next_neighbor = "No measurement";
        display_data_msg_.mean = "No measurement";
        display_data_msg_.min = "No measurement";
        display_data_msg_.max = "No measurement";
        display_data_pub_->publish(display_data_msg_);
      } else {
        std::vector<uint64_t> points_in_area =
          inspection_->getSparseMeasurementsInRadius(point_coords, selection_sphere_radius_);
        uint64_t closests_point = inspection_->getClosestSparseMeasurement(point_coords);
        std::vector<double> values_in_area =
          inspection_->getSparseValuesForIds(params_.sparse.value_to_display, points_in_area);
        std::vector<std::string> units = inspection_->getSparseUnits();
        mtx_.unlock();

        double mean_value = 0;
        double min_value = 0;
        double max_value = 0;
        std::string unit = units[0];  // todo the index 0 should be chosen dynamically
        if (values_in_area.size() > 0) {
          mean_value = mean(values_in_area);
          auto minmax = std::minmax_element(
            values_in_area.begin(), 
            values_in_area.end()
          );
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
  void denseDataReq(std_msgs::msg::Empty)
  {
    mtx_.lock();
    if (inspection_->getDenseDataCount() == 0) {
      mtx_.unlock();
      RCLCPP_INFO(this->get_logger(), "No dense data available");
    } else {
      RCLCPP_INFO(this->get_logger(), "Asking vinspect for images");
      int id_to_get = inspection_->getClosestDenseMeasurement(dense_interactive_marker_pose_);
      auto image = inspection_->getImageFromId(id_to_get);
      auto pose = inspection_->getDensePoseFromId(id_to_get);
      pubRefMeshDense(vinspect::eulerToQuatPose(pose));
      auto msg = cv_bridge::CvImage(
        std_msgs::msg::Header(), // We do not have any header information at this point
        "rgb8",
        image
      ).toImageMsg();
      dense_image_pub_->publish(*msg);
      RCLCPP_INFO(this->get_logger(), "Published image");
      mtx_.unlock();
    }
  }

  double roundValue(double value)
  {
    if (params_.round_to_decimals < 0) {
      // no rounding
      return value;
    } else {
      // round to number of decimals after the decimal point
      return round(value * pow(10, params_.round_to_decimals)) / pow(10, params_.round_to_decimals);
    }
  }

  void multiDenseDataReq(std_msgs::msg::Int32 msg)
  {
    std::vector<std::array<double, 6>> poses = inspection_->getMultiDensePoses(msg.data);

    visualization_msgs::msg::MarkerArray markerArr = visualization_msgs::msg::MarkerArray();
    for (size_t i = 0; i < poses.size(); i++) {
      std::array<double, 7> quat_pose = vinspect::eulerToQuatPose(poses[i]);
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
    marker.header.frame_id = params_.frame_id;
    marker.type = 10;
    marker.id = 0;
    marker.ns = "ref_mesh";
    marker.action = marker.DELETE;
    // Delete the old marker first, out of some reason, MODIFY does not work
    ref_marker_pub_->publish(marker);

    marker.action = marker.ADD;
    marker.frame_locked = true;
    marker.mesh_resource = params_.ref_mesh_path;
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
      inspection_->clear();
      sparse_mesh_->resetMesh();
      mesh_marker_msg_.points.clear();
      mesh_marker_msg_.colors.clear();
      sparse_mesh_pub_->publish(mesh_marker_msg_);
      mtx_.unlock();
    }
    settings_changed_ = true;
  }

  void cameraInfoCb(const sensor_msgs::msg::CameraInfo msg, const std::string sensor_name)
  {
    // todo maybe we should provide them at the construction of the inspection object
    //  todo need to differentiate between different cameras
    open3d::camera::PinholeCameraIntrinsic intrinsic = open3d::camera::PinholeCameraIntrinsic(
      msg.width, msg.height, msg.k[0], msg.k[4], msg.k[2], msg.k[5]);
    inspection_->setIntrinsic(intrinsic, 0); // TODO make sensor id string
  }

  void cameraCb(
    const sensor_msgs::msg::Image::ConstSharedPtr & color_image_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_image_msg,
    const std::string sensor_name
  ) {
    if (!dense_pause_) {
      // Get sensor specific parameters
      auto sensor_params = params_.dense_sensor_names_map.at(sensor_name);


      // color needs to be rgb8
      if(color_image_msg->encoding != "rgb8" && color_image_msg->encoding != "bgr8") {
        RCLCPP_ERROR(this->get_logger(), "Unsupported encoding: %s", color_image_msg->encoding.c_str());
        return;
      }
      //  Convert ROS image message to OpenCV
      cv_bridge::CvImageConstPtr cv2_color_img, cv2_depth_img;
      try {
        cv2_color_img = cv_bridge::toCvShare(color_image_msg, "rgb8");
        // we keep depth in the given format to not loose precision
        cv2_depth_img = cv_bridge::toCvShare(depth_image_msg, "");
      } catch (cv_bridge::Exception & e) {
        RCLCPP_ERROR(this->get_logger(), "Error converting image from ROS to CV");
        return;
      }

      geometry_msgs::msg::TransformStamped transformed_pose_optical;
      geometry_msgs::msg::TransformStamped transformed_pose_world;
      try {
        transformed_pose_optical = tf_buffer_->lookupTransform(
          sensor_params.optical_frame_id, params_.frame_id, color_image_msg->header.stamp, 100ms);

        /* Note: It is expected that an equivalent non-optical frame exists
        to the optical frame in which the image is published. */
        transformed_pose_world = tf_buffer_->lookupTransform(
          params_.frame_id, sensor_params.frame_id, color_image_msg->header.stamp, 100ms);
      } catch (tf2::TransformException & e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get transform: %s", e.what());
        return;
      }

      Eigen::Matrix4d rgb_pose_tsdf = transformStampedToTransformMatix(transformed_pose_optical);
      Eigen::Matrix4d rgb_pose_world = transformStampedToTransformMatix(transformed_pose_world);


      inspection_->addImage(
        cv2_color_img->image,
        cv2_depth_img->image,
        depth_trunc_,
        std::stoi(sensor_name), 
        rgb_pose_tsdf, 
        rgb_pose_world);
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
      depth_trunc_ = request->depth_trunc;
      inspection_->reinitializeTSDF(request->voxel_length);
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

  std::unique_ptr<vinspect::Inspection> inspection_{nullptr};
  std::unique_ptr<vinspect::SparseMesh> sparse_mesh_{nullptr};

  double old_transparency_ = 0;
  uint64_t last_mesh_number_sparse_ = 0;
  double dot_size_ = 0;
  double selection_sphere_radius_ = 0;
  int mean_min_max_;
  bool use_custom_color_ = false;
  bool paused_ = false;
  bool dense_pause_ = true;
  bool settings_changed_ = false;
  std::mutex mtx_;
  visualization_msgs::msg::Marker mesh_marker_msg_;
  vinspect_msgs::msg::AreaData display_data_msg_;
  vinspect_msgs::msg::Status status_msg_;

  double depth_trunc_ = 3.0;

  std::array<double, 7> dense_interactive_marker_pose_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};

  std::shared_ptr<vinspect::ParamListener> param_listener_{nullptr};
  vinspect::Params params_;

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
    selection_marker_sub_, pose_marker_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr dense_req_sub;
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
  rclcpp::shutdown();
}
