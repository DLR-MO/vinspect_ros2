# Vinspect ROS2

This package provides the ROS2 interface for the [Vinspect software](https://github.com/DLR-MO/vinspect).

![example image of a reconstruction](./doc/reconstruction.PNG)
Example of an inspection with RGB-D data from a robot.

![example image of an ultrasonic inspection](./doc/ultrasonic.png)
Example of an inspection with a handheld ultrasonic sensor, using the setup described in Wilken et al. "Localisation of Ultrasonic NDT Data Using Hybrid Tracking of Component and Probe".

## Installation
There are four colcon packages that you need to build:
- [vinspect](https://github.com/DLR-MO/vinspect) (the core of vinspect)
- [vinspect_msgs](https://github.com/DLR-MO/vinspect_msgs) (the message definitions)
- [vinspect_rviz_plugins](https://github.com/DLR-MO/vinspect_rviz_plugins) (RViz2 plugins for better visualisation)
- [vinspect_ros2](https://github.com/DLR-MO/vinspect_ros2) (the ROS2 interface)

Clone these packages in your colcon workspace (beware that the `vinspect` package has a git submodule that might need to intantiate) and call `colcon build`.

## Usage
This package provides a node that you can launch with specifying mutliple parameters (see below).
By setting these parameters, it is possible to subscribe on multiple different sensor inputs.
These will then be aggregated and visualized using RViz2 with additional plugins (see below).
For more details on the vinspect software itself, see the [Vinspect package](https://github.com/DLR-MO/vinspect).

### Recording
Currently, two different types of sensor inputs are supported:
- `vinspect_msgs/Sparse` messages 
- `RGB-D` images (via color and depth `sensor_msgs/Image` messages)
  
For the `Sparse` messages, it is expected that the user implements an additional preprocessing node 
that takes in some sort of point meassurement (e.g. from a ultrasonic sensor) and connects it to the
pose in which the measurement was taken.
Additionally, each sensor needs to have a unique ID.
The message can contain multiple data entries and also a custom color in which this data should be displayed.
This approach was taken since there is no standard message for all types of sensor inputs.
As soon as the Vinspect node is started, it will record these messages.

The `RGB-D` messages are only recorded after calling the `/vinspect/start_reconstruction` service
to prevent recording images before the camera is at the correct position.
This service can also be called with the Vinspect settings RViz2 plugin.
It is possible to use multiple cameras at the same time.

### Visualization
The visualization is integrated into RViz2.
Both, the sparse measurement visualization and the 3D reconstruction based on RGB-D data, are 
published as `visualization_msgs/Marker` messages.
They can be displayed with the standard RViz marker visualization.
Additionally, the [vinspect_rviz_plugins](https://github.com/DLR-MO/vinspect_rviz_plugins) package 
provides three plugins for RViz2:
- a settings plugin which allows you to configure the visualization of both sparse and dense data
- a status panel which provides the current status of the inspection, e.g. how many measurements 
  have been taken
- an area plugin which works together with an interactive marker which allows you to investigate
  measurements at a certain location. The interactive marker needs to launched with 
  `ros2 run vinspect_ros2 selection_marker.py` (or from your launch file)

### Parameters

See [vinspect_parameters.yaml](src/vinspect_parameters.yaml) for the list of parameters including descriptions, valid ranges, etc..

## Demo
You can get a better impression on how this package works with the following demos.

### Sparse Data
Launch the inspection:
`ros2 launch vinspect_ros2 sparse_demo.launch.py`

You should now be presented with the RViz GUI including the special Vinspect plugins (on the right side).
There should be a reference box visible, if not try adjusting the transparency on the right.

Start a dummy publisher:
`ros2 run vinspect_ros2 dummy_publisher.py`

Now randomly generated measurements should appear in form of octagons.

Watch how the dummy data is recieved and saved concurrently:
`watch -n 0.1 tail -n 10 /tmp/demo.vinspect`

Stop the software by pressing `Ctrl+C`.

Run again the same command to load the previous data and allow continuing the inspection:
`ros2 launch vinspect_ros2 sparse_demo.launch.py`

You can move the interactive marker to investigate the measurement values.
You can adapt settings in the settings panel.

### Dense Data
You can see how the software is started in the `dense_demo.launch.py` file.

**TODO provide link to a rosbag to allow actually reconstructing**

# Contributing
Contributions are welcome (issues as well as pull requests).
We plan on further developing and using this software in the future and are happy for any 
contributions from the community.
We follow the [ROS2 code style](https://docs.ros.org/en/rolling/The-ROS2-Project/Contributing/Code-Style-Language-Versions.html).