#! /usr/bin/env python3

"""Script to load rosbags directly into Vinspect."""
import faulthandler
from datetime import datetime
import importlib
from pathlib import Path
from pprint import pprint
import sys
import tempfile

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image

import numpy as np

import open3d as o3d

from tqdm import tqdm

from rclpy.duration import Duration
from rclpy.time import Time

from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from rosbags.typesys import Stores, get_typestore

from tf2_ros import Buffer

from transforms3d.affines import compose
from transforms3d.quaternions import quat2mat

from vinspect.vinspect_py import Inspection, DenseSensor, add_image_py

import yaml


faulthandler.enable()
TYPESTORE = get_typestore(Stores.LATEST)

TF_TOPIC = '/tf'
TF_STATIC_TOPIC = '/tf_static'

DEPTH_TRUNC = 2.0  # m

NATIVE_CLASSES: dict[str, type] = {}

# Used as described here in the official documentation
# https://ternaris.gitlab.io/rosbags/examples/use_with_native.html


def to_native(msg: object) -> object:
    """Convert rosbags message to native message.

    Args:
        msg: Rosbags message.

    Returns:
        Native message.

    """
    if isinstance(msg, list):
        return [to_native(item) for item in msg]

    msgtype: str = msg.__msgtype__  # type: ignore[attr-defined]
    if msgtype not in NATIVE_CLASSES:
        pkg, name = msgtype.rsplit('/', 1)
        NATIVE_CLASSES[msgtype] = getattr(
            importlib.import_module(pkg.replace('/', '.')), name)

    fields = {}
    # type: ignore[attr-defined]
    for name, field in msg.__dataclass_fields__.items():
        if 'ClassVar' in field.type:
            continue
        value = getattr(msg, name)
        if '__msg__' in field.type:
            value = to_native(value)
        elif isinstance(value, list):
            value = [to_native(x) for x in value]
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        fields[name] = value

    return NATIVE_CLASSES[msgtype](**fields)


def read_tf(bag_path, topic_message_numbers, args):
    """Read tf messages into a tf buffer so that we can later compute transformations."""
    with AnyReader([Path(bag_path)], default_typestore=TYPESTORE) as reader:
        # First read all tf and tf_static messages into the buffer
        # Create tf2 buffer with a buffer length of the whole rosbag
        length_ns = reader.duration
        buffer = Buffer(Duration(seconds=length_ns/1e9,
                        nanoseconds=length_ns % 1e9))
        # Fill with data
        print('Reading static TF messages')
        connections = [
            x for x in reader.connections if x.topic == TF_STATIC_TOPIC]
        for connection, _, rawdata in tqdm(
            reader.messages(connections=connections),
            total=topic_message_numbers[TF_STATIC_TOPIC]
        ):
            msg = to_native(reader.deserialize(rawdata, connection.msgtype))
            for transform in msg.transforms:
                buffer.set_transform_static(transform, 'rosbag_reader')

        print('\nReading dynamic TF messages')
        connections = [x for x in reader.connections if x.topic == TF_TOPIC]
        for connection, _, rawdata in tqdm(
            reader.messages(connections=connections), 
            total=topic_message_numbers[TF_TOPIC]
        ):
            msg = to_native(reader.deserialize(rawdata, connection.msgtype))
            # each message can contain multiple transforms
            for transform in msg.transforms:
                buffer.set_transform(transform, 'rosbag_reader')
    return buffer


def read_camera_infos(bag_path: str, inspection: Inspection, info_topics: list[str]):
    """Read the camera messages and set the intrinsics for the sensors."""
    with AnyReader([Path(bag_path)], default_typestore=TYPESTORE) as reader:
        # read camera infos one by one to make sure we match sensor IDs correctly
        print('Reading camera info messages')
        for sensor, camera_topic in enumerate(info_topics):
            connections = [
                x for x in reader.connections if x.topic == camera_topic]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                inspection.set_intrinsic2(sensor, msg.width, msg.height,
                                          msg.k[0], msg.k[4], msg.k[2], msg.k[5])
                break
        print('Finished reading camera info messages')


def get_affine_matrix_from_tf(buffer, frame, stamp, inspection_frame):
    """Compute affinity matrix from tf message."""
    try:
        trans = buffer.lookup_transform(frame, inspection_frame, 
            Time(seconds=stamp.sec, nanoseconds=stamp.nanosec))
    except Exception as e:
        print(e)
        print('Ignored image that could not be transformed')
        return
    t = trans.transform
    return compose(np.array([t.translation.x, t.translation.y, t.translation.z]),
                   quat2mat(
                       np.array([t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z])),
                   np.array([1.0, 1.0, 1.0]))


def integrate(des_color, des_depth, sensor_id, buffer, inspection, fixed_frame):
    """Integrate rgbd image into TSDF with corresponding transformation."""
    # get corresponding optical pose from tf2
    # TODO use a better rosbag with correct camera info...
    frame="camera1_depth_optical_frame" 

    affine_matrix_optical = get_affine_matrix_from_tf(
        buffer, frame, des_depth.header.stamp, fixed_frame)
    
    if affine_matrix_optical is None:
        return

    # get the corresponding camera pose (for retrival of camera poses) by guessing the name
    # TODO use better method to get frames of cameras
    affine_matrix_camera = get_affine_matrix_from_tf(
        buffer, frame, des_depth.header.stamp, fixed_frame)
    
    if affine_matrix_optical is not None and affine_matrix_camera is not None:
        add_image_py(inspection,
                     message_to_cvimage(des_color),
                     message_to_cvimage(des_depth), DEPTH_TRUNC,
                     sensor_id, affine_matrix_optical, affine_matrix_camera)


def read_images(bag_path, inspection, topic_to_id, topic_message_numbers, tf_buffer, args):
    """Read in all images in the bag and match corresponding color and depth images."""
    with AnyReader([Path(bag_path)], default_typestore=TYPESTORE) as reader:
        print('Reading image messages')
        # We are reading the messages in topic pairs to match color and depth
        for sensor_id in range(len(args.color_topics)):
            open_color_images: list[Image] = []
            open_depth_images: list[Image] = []
            connections = [
                x for x in reader.connections if x.topic in [args.color_topics[sensor_id],
                                                             args.depth_topics[sensor_id]]]
            matched_images = 0
            print(
                f'Reading images of sensor {sensor_id} of {len(args.color_topics)}')

            num_img_msgs = topic_message_numbers[args.color_topics[sensor_id]] + \
                topic_message_numbers[args.depth_topics[sensor_id]]

            for connection, _, rawdata in tqdm(
                reader.messages(connections=connections),
                total=num_img_msgs
            ):
                msg: Image = to_native(
                    reader.deserialize(rawdata, connection.msgtype))
                sensor_id = topic_to_id[connection.topic]
                integrated = False
                if connection.topic in args.color_topics:
                    for idx, depth_image in enumerate(open_depth_images):
                        depth_time = Time.from_msg(depth_image.header.stamp)
                        color_time = Time.from_msg(msg.header.stamp)
                        if abs((depth_time - color_time).nanoseconds)/1e9 < args.max_time_delta:
                            integrate(msg, depth_image, sensor_id,
                                      tf_buffer, inspection, args.fixed_frame)
                            integrated = True
                            matched_images += 1
                            open_depth_images.pop(idx)
                            break
                    if not integrated:
                        open_color_images.append(msg)
                else:
                    for idx, color_image in enumerate(open_color_images):
                        depth_time = Time.from_msg(msg.header.stamp)
                        color_time = Time.from_msg(color_image.header.stamp)
                        if abs((depth_time - color_time).nanoseconds)/1e9 < args.max_time_delta:
                            integrate(color_image, msg, sensor_id,
                                      tf_buffer, inspection, args.fixed_frame)
                            integrated = True
                            matched_images += 1
                            open_color_images.pop(idx)
                            break
                    if not integrated:
                        open_depth_images.append(msg)

            print(f'{matched_images} images were matched correctly.')
            print(f'The following count of images were not matchable for sensor {sensor_id}\n  color: \
                  {len(open_color_images)}\n  depth: {len(open_depth_images)}')


def process_bag(bag_path, args):
    """Create a Vinspect object and read the neccessary topics in the bag."""
    # Create sensor objects
    assert len(args.color_topics) == len(args.depth_topics) and \
        len(args.color_topics) == len(args.info_topics) and \
        len(args.color_topics) == len(args.widths) and \
        len(args.color_topics) == len(args.heights) and \
        len(args.color_topics) == len(args.depth_scales), \
        'The provided sensor attributes do not match in length'

    dense_sensors: DenseSensor = []
    for i, (width, height, depth_scale) in enumerate(zip(
        args.widths,
        args.heights,
        args.depth_scales
    )):
        dense_sensors.append(
            DenseSensor(i, width, height, depth_scale)
        )

    # Create a Vinspect object
    inspection = Inspection([],
                            dense_sensors,
                            save_path=tempfile.mktemp(),
                            inspection_space_3d_min=args.inspection_space_min,
                            inspection_space_3d_max=args.inspection_space_max)
    inspection.reinitialize_TSDF(args.voxel_length)

    topic_to_id = {}
    for i, (t1, t2) in enumerate(zip(args.color_topics, args.depth_topics)):
        topic_to_id[t1] = i
        topic_to_id[t2] = i

    try:
        with AnyReader([Path(bag_path)], default_typestore=TYPESTORE) as reader:
            # Check if all topics are present
            for topic in \
                    args.color_topics + \
                    args.depth_topics + \
                    args.info_topics + \
                    [TF_TOPIC] + \
                    [TF_STATIC_TOPIC]:
                exists_in_bag = any(
                    connection.topic == topic for connection in reader.connections
                )
                if not exists_in_bag:
                    print(f'Could not find requested topic {topic} in bag. \
                           Following topics are in the bag:')
                    for connection in reader.connections:
                        print(connection.topic, connection.msgtype)
                    sys.exit(1)

            # Get number of message on the used topics for progressbar later
            topic_message_numbers = {}
            with open(bag_path+'/metadata.yaml') as f:
                meta_data = yaml.safe_load(f)
            for topic_data in meta_data['rosbag2_bagfile_information']['topics_with_message_count']:
                topic_message_numbers[topic_data['topic_metadata']
                                      ['name']] = topic_data['message_count']
            print(f'Using bag with messages from {datetime.fromtimestamp(reader.start_time/1e9)} \
                  to {datetime.fromtimestamp(reader.end_time/1e9)} with a duration of \
                    {(reader.end_time-reader.start_time)/1e9} s')
            pprint(topic_message_numbers)

            # Sanity check
            for camera_topic in args.info_topics:
                if topic_message_numbers.get(camera_topic, 0) == 0:
                    print(f'Did not find camera info messages for all cameras! Problem with \
                          camera_info topic {camera_topic}')

    except UnicodeDecodeError:
        print('Error decoding bag file. Are you sure that you provided the path to the folder and \
              not to the mcap file?')

    tf_buffer = read_tf(bag_path, topic_message_numbers, args)
    read_camera_infos(bag_path, inspection, args.info_topics)
    read_images(bag_path, inspection, topic_to_id,
                topic_message_numbers, tf_buffer, args)

    # Provide statistics to the user when finished reading the bag
    print('Statistics:')
    print(f'Integrated images {inspection.get_integrated_images_count()}')
    if inspection.get_integrated_images_count() == 0:
        print('No mesh could be reconstructed. Please check if the inspection space boundaries \
              are correct.')
    else:
        # save the mesh
        inspection.save_dense_reconstruction(args.output_mesh_path)
        print(f'Mesh saved to {args.output_mesh_path}')
        mesh = o3d.io.read_triangle_mesh(args.output_mesh_path)
        o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process a ROS2 bag file.')
    parser.add_argument('bag_path', help='Path to the ROS2 bag file.')
    parser.add_argument('output_mesh_path',
                        help='Path to save the mesh file (.ply).')
    parser.add_argument('--color-topics', type=str, nargs='+',
                        help='The color topics that should be used')
    parser.add_argument('--depth-topics', type=str, nargs='+',
                        help='The depth topics that should be used, in matching order to the color \
                            topics')
    parser.add_argument('--info-topics', type=str, nargs='+',
                        help='The camera info topics that should be used, in matching order to the \
                            color topics')
    parser.add_argument('--widths', type=int, nargs='+',
                        help='The image widths that should be used, in matching order to the color \
                            and depth topics')
    parser.add_argument('--heights', type=int, nargs='+',
                        help='The image heights that should be used, in matching order to the color \
                            and depth topics')
    parser.add_argument('--depth-scales', type=float, nargs='+',
                        help='The depth scales that should be used, in matching order to the depth \
                            topics')
    parser.add_argument('--voxel-length', type=float,
                        default=0.005, help='Voxel length [m]')
    parser.add_argument('--inspection-space-min', type=float, nargs='+', default=[-1.0, -1.0, -1.0],
                        help='Inspection space minimal boundaries as vector x y z in m')
    parser.add_argument('--inspection-space-max', type=float, nargs='+', default=[1.0, 1.0, 1.0],
                        help='Inspection space maximal boundaries as vector x y z in m')
    parser.add_argument('--fixed-frame', type=str, default='world',
                        help='Fixed frame for the inspection space')
    parser.add_argument('--max-time-delta', type=float, default=0.1,
                        help='Maximum temporal difference between matched rgb and color images')
    args = parser.parse_args()
    try:
        process_bag(args.bag_path, args)
    except KeyboardInterrupt:
        print("Stopped by user")