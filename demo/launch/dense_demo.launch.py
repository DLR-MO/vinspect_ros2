# SPDX-FileCopyrightText: 2024 Marc Bestmann <marc.bestmann@dlr.de>
#
# SPDX-License-Identifier: MIT

import os.path

from ament_index_python.packages import get_package_share_directory
import launch
import launch_ros.actions


def generate_launch_description():
    vinspect_node = launch_ros.actions.Node(
        package='vinspect_ros2',
        executable='node',
        name='vinspect_ros2',
        output='screen',
        emulate_tty=True,
        parameters=[
            {
                'frame_id': 'world',
                'sensor_types': ['RGBD'],
                'rgbd_color_topics': [
                    '/raw/left/image',
                ],
                'rgbd_depth_topics': [
                    '/depth/image',
                ],
                'rgbd_info_topics': [
                    '/raw/left/camera_info',
                ],
                'use_sim_time': False,
                'inspection_space_3d_min': [-1.0, -1.0, -1.0],
                'inspection_space_3d_max': [1.0, 1.0, 1.0],
                'inspection_space_6d_min': [-100.0, -100.0, -100.0, -20.0, -20.0, -20.0],
                'inspection_space_6d_max': [100.0, 100.0, 100.0, 20, 20, 20],
                # 'save_path': '/tmp/demo_dense.vinspect',
                'dense_senor_resolution': [2472.0, 2064.0],
            }
        ],
    )

    rviz = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        emulate_tty=True,
        arguments=['-d' + os.path.join(get_package_share_directory(
            'vinspect_ros2'), 'config', 'dense_demo.rviz')],
    )

    pose_marker_node = launch_ros.actions.Node(
        package='vinspect_ros2',
        executable='pose_marker.py',
        name='pose_marker',
        output='screen',
    )

    static_frame_node = launch_ros.actions.Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf2_broadcaster",
        arguments=['--frame-id', '/world', '--child-frame-id', '/base_link'],
    )

    return launch.LaunchDescription(
        [
            vinspect_node,
            rviz,
            pose_marker_node,
            static_frame_node,
        ]
    )
