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
        #prefix="gdbserver localhost:3000",
        parameters=[
            {
                'use_sim_time': False,
                'frame_id': 'world',
                'save_path': '/tmp/demo_dense.vinspect',
                'round_to_decimals': 2,
                'dense_sensor_names': ["0"],  # TODO string or int sensor name...
                '0': {
                    'color_topic': '/raw/left/image',
                    'depth_topic': '/depth/image',
                    'camera_info_topic': '/raw/left/camera_info',
                    'width': 2472,
                    'height': 2064,
                    'depth_scale': 1.0
                },
                'inspection_space_3d': {
                    'min': [-1.0, -1.0, -1.0],
                    'max': [1.5,  1.0,  1.0]
                },
                'inspection_space_6d': {
                    'min': [-100.0, -100.0, -100.0, -20.0, -20.0, -20.0],
                    'max': [100.0, 100.0, 100.0, 20, 20, 20]
                },
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
