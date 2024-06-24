import os.path
from ament_index_python.packages import get_package_share_directory
import launch
import launch_ros.actions
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    realsense_camera_node = IncludeLaunchDescription(PythonLaunchDescriptionSource(get_package_share_directory(
        'realsense2_camera')+"/launch/rs_launch.py"), launch_arguments={"rgb_camera.profile": "848x480x30"}.items())

    static_transform_node = launch_ros.actions.Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["--x", "0.039313", "--y", "-0.041174", "--z", "-0.060189", "--qx", "-0.3067751", "--qy", "-0.0678829",
                   "--qz", "-0.3779223", "--qw", "0.8708936", "--frame-id", "marker", "--child-frame-id", "handheld"]
    )

    static_transform_node2 = launch_ros.actions.Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["--x", "0.0", "--y", "-0.0", "--z", "-0.0", "--yaw",
                   "2.355", "--frame-id", "handheld", "--child-frame-id", "camera_link"]
    )

    markerbased_tracking_node = launch_ros.actions.Node(
        package="markerbased_tracking",
        executable="markerbased_tracking",
        name="markerbased_tracking",
        output="screen",
        emulate_tty=True,
        respawn=True,
        respawn_delay=0.0,
        parameters=[{
            "frame": "world",
            "child_frame": "marker",
            "ip": "192.168.0.2",
        }
        ]
    )

    vinspect_node = launch_ros.actions.Node(
        package="vinspect_ros2",
        executable="node",
        name="vinspect_ros2",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "frame_id": "world",
                "sensor_types": ["RGB"],
                "rgbd_color_topics": [
                    "/camera/camera/color/image_raw",
                ],
                "rgbd_depth_topics": [
                    "/camera/camera/depth/image_rect_raw",
                ],
                "rgbd_info_topics": [
                    "/camera/camera/color/camera_info",
                ],
                "inspection_space_3d_min": [-0.0, -0.0, -1.0],
                "inspection_space_3d_max": [4.0, 4.0, 4.0],
            }
        ],
    )

    rviz = launch_ros.actions.Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        emulate_tty=True,
        arguments=["-d" + os.path.join(get_package_share_directory(
            "vinspect_ros2"), "config", "dense_demo.rviz")],
    )

    return launch.LaunchDescription(
        [
            static_transform_node,
            static_transform_node2,
            realsense_camera_node,
            markerbased_tracking_node,
            vinspect_node,
            rviz,
        ]
    )
