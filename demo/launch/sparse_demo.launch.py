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
        #prefix="gdbserver localhost:3000",
        parameters=[
            {
                'ref_mesh_path': 'package://vinspect_ros2/data/box.ply',
                'frame_id': 'world',
                'save_path': '/tmp/demo.vinspect',
                'round_to_decimals': 2,
                'sparse': {
                    'value_to_display': 'random_demo',
                    'value_names':  ['random_demo'],
                    'value_units': ['no_unit'],#
                    'topic': "sparse",
                }
            }
        ],
    )

    interactive_marker_node = launch_ros.actions.Node(
        package='vinspect_ros2',
        executable='selection_marker.py',
        name='selection_marker',
        output='screen',
    )

    static_frame_node = launch_ros.actions.Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf2_broadcaster",
        arguments=['--frame-id', '/world', '--child-frame-id', '/base_link'],
    )

    rviz = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d' + os.path.join(get_package_share_directory(
            'vinspect_ros2'), 'config', 'demo.rviz')],
    )

    return launch.LaunchDescription(
        [
            vinspect_node,
            interactive_marker_node,
            static_frame_node,
            rviz,
        ]
    )
