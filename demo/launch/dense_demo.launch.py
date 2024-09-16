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
                'sensor_types': ['RGBD', 'RGBD', 'RGBD'],
                'rgbd_color_topics': [
                    '/camera/camera/color/image_rect_raw',
                    '/camera1/camera1/color/image_rect_raw',
                    '/camera2/camera2/color/image_rect_raw',
                ],
                'rgbd_depth_topics': [
                    '/camera/camera/depth/image_rect_raw',
                    '/camera1/camera1/depth/image_rect_raw',
                    '/camera2/camera2/depth/image_rect_raw',
                ],
                'rgbd_info_topics': [
                    '/camera/camera/color/camera_info',
                    '/camera1/camera1/color/camera_info',
                    '/camera2/camera2/color/camera_info',
                ],
                'use_sim_time': True,
                'inspection_space_min': [-2.5, -2.0, -2.0],
                'inspection_space_max': [-1.5, 2.0, 2.0],
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
            'vinspect_ros2'), 'config', 'demo.rviz')],
    )

    return launch.LaunchDescription(
        [
            vinspect_node,
            rviz,
        ]
    )
