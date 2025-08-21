"""Script to load rosbags directly into Vinspect."""
import faulthandler
from datetime import datetime
from pathlib import Path
from pprint import pprint

from geometry_msgs.msg import TransformStamped

import numpy as np

import open3d as o3d

from rclpy.duration import Duration
from rclpy.time import Time

from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from rosbags.image.image import to_cvtype
from rosbags.typesys import Stores, get_typestore

from tf2_ros import Buffer

from transforms3d.affines import compose
from transforms3d.quaternions import quat2mat

from vinspect.vinspect_py import Inspection, integrate_image_py

import yaml


faulthandler.enable()
TYPESTORE = get_typestore(Stores.LATEST)

TF_TOPIC = '/tf'
TF_STATIC_TOPIC = '/tf_static'

DEPTH_SCALE = 1000.0
DEPTH_TRUNC = 0.50  # m
WORLD_LINK = 'world'
# this is only for debugging time issues in the messages
TIME_OFFSET = 0.0  # 0.99*1e9 #ns
# maximal delta time for color and depth message in seconds
MAX_TIME_DELTA = 0.01  # seconds


def deserialize_tf_to_msg(desirialized_msg):
    """
    Deserilize a transformation to a message.

    The rosbag library does not provide the real message object, so we need to create it manually
    """
    transform = TransformStamped()
    transform.header.frame_id = desirialized_msg.header.frame_id
    transform.header.stamp = desirialized_msg.header.stamp
    transform.child_frame_id = desirialized_msg.child_frame_id
    transform.transform.translation.x = desirialized_msg.transform.translation.x
    transform.transform.translation.y = desirialized_msg.transform.translation.y
    transform.transform.translation.z = desirialized_msg.transform.translation.z
    transform.transform.rotation.x = desirialized_msg.transform.rotation.x
    transform.transform.rotation.y = desirialized_msg.transform.rotation.y
    transform.transform.rotation.z = desirialized_msg.transform.rotation.z
    transform.transform.rotation.w = desirialized_msg.transform.rotation.w
    return transform


def read_tf(bag_path, topic_message_numbers, args):
    """Read tf messages into a tf buffer so that we can later compute transformations."""
    read_message = 0
    with AnyReader([Path(bag_path)], default_typestore=TYPESTORE) as reader:
        # First read all tf and tf_static messages into the buffer
        # Create tf2 buffer with a buffer length of the whole rosbag
        length_ns = reader.duration
        buffer = Buffer(Duration(seconds=length_ns/1e9, nanoseconds=length_ns % 1e9))
        # Fill with data
        print('Reading static TF messages')
        connections = [x for x in reader.connections if x.topic == TF_STATIC_TOPIC]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = TYPESTORE.deserialize_cdr(rawdata, connection.msgtype)
            for deserizled_transform in msg.transforms:
                buffer.set_transform_static(deserialize_tf_to_msg(
                    deserizled_transform), 'rosbag_reader')
            read_message += 1
            if (read_message) % 1 == 0:
                print(f'\rProcessing... \
                      {read_message / topic_message_numbers[TF_STATIC_TOPIC] * 100:.2f}% done',
                      end='')

        if args.apply_tf_hack:
            hack_transform = TransformStamped()
            hack_transform.header.frame_id = 'camera2_color_optical_frame'
            hack_transform.child_frame_id = 'camera2_color_optical_frame2'
            hack_transform.transform.translation.x = 0
            hack_transform.transform.translation.y = 0
            hack_transform.transform.translation.z = 0
            hack_transform.transform.rotation.x = 0
            hack_transform.transform.rotation.y = 0
            hack_transform.transform.rotation.z = 1
            hack_transform.transform.rotation.w = 0
            buffer.set_transform_static(hack_transform, 'hack_transform')

        read_message = 0
        print('\nReading dynamic TF messages')
        connections = [x for x in reader.connections if x.topic == TF_TOPIC]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = TYPESTORE.deserialize_cdr(rawdata, connection.msgtype)
            # each message can contain multiple transforms
            for deserizled_transform in msg.transforms:
                buffer.set_transform(deserialize_tf_to_msg(deserizled_transform), 'rosbag_reader')
            read_message += 1
            if (read_message) % 1000 == 0:
                percentage_done = read_message / topic_message_numbers[TF_TOPIC] * 100
                print(f'\rProcessing... {percentage_done: .2f}% done', end='')
                if percentage_done > args.read_percentage:
                    break
        print('\rProcessing... 100.00% done')
    return buffer


def read_camera_infos(bag_path, inspection, args):
    """Read the camera messages and set the intrinsics for the sensors."""
    sensor_id = 0
    with AnyReader([Path(bag_path)], default_typestore=TYPESTORE) as reader:
        # read camera infos one by one to make sure we match sensor IDs correctly
        print('Reading camera info messages')
        for camera_topic in args.info_topics:
            connections = [x for x in reader.connections if x.topic == camera_topic]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = TYPESTORE.deserialize_cdr(rawdata, connection.msgtype)
                inspection.set_intrinsic2(sensor_id, msg.width, msg.height,
                                          msg.k[0], msg.k[4], msg.k[2], msg.k[5])
                break
            sensor_id += 1
        print('Finished reading camera info messages')


def get_affine_matrix_from_tf(buffer, frame, stamp):
    """Compute affinity matrix from tf message."""
    try:
        if args.apply_tf_hack:
            if frame == 'camera2_color_optical_frame':
                frame = 'camera2_color_optical_frame2'
        trans = buffer.lookup_transform(frame, WORLD_LINK, Time(
            seconds=stamp.sec, nanoseconds=stamp.nanosec) - Time(nanoseconds=TIME_OFFSET))
    except Exception as e:
        print(e)
        print('Ignored image that could not be transformed')
        return
    t = trans.transform
    return compose(np.array([t.translation.x, t.translation.y, t.translation.z]),
                   quat2mat(np.array([t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z])),
                   np.array([1.0, 1.0, 1.0]))


def integrate(des_color, des_depth, sensor_id, buffer, inspection):
    """Integrate rgbd image into TSDF with corresponding transformation."""
    # get corresponding optical pose from tf2
    # TODO check if it makes sense that we use the color msg as frame of reference
    affine_matrix_optical = get_affine_matrix_from_tf(
        buffer, des_color.header.frame_id, des_color.header.stamp)
    # get the corresponding camera pose (for retrival of camera poses) by guessing the name
    # TODO use better method to get frames of cameras
    frame = des_color.header.frame_id[:-20] + '_link'
    affine_matrix_camera = get_affine_matrix_from_tf(buffer, frame, des_color.header.stamp)

    if affine_matrix_optical is not None and affine_matrix_camera is not None:
        color_depth, color_fmt, color_typestr, color_nchan = to_cvtype(des_color.encoding)
        depth_depth, depth_fmt, depth_typestr, depth_nchan = to_cvtype(des_depth.encoding)
        if color_typestr == 'uint8':
            color_str = '8U'
        else:
            print(f'No method to handle color image of type {color_depth} {color_fmt} \
                  {color_typestr} {color_nchan}')
            exit()
        if depth_typestr == 'uint16':
            depth_str = '16U'
        else:
            print(f'No method to handle depth image of type {depth_depth} {depth_fmt} \
                  {depth_typestr} {depth_nchan}')
            exit()
        integrate_image_py(inspection, message_to_cvimage(des_color), color_str,
                           message_to_cvimage(des_depth), depth_str, DEPTH_SCALE, DEPTH_TRUNC,
                           sensor_id, affine_matrix_optical, affine_matrix_camera)


def read_images(bag_path, inspection, topic_to_id, topic_message_numbers, tf_buffer, args):
    """Read in all images in the bag and match corresponding color and depth images."""
    with AnyReader([Path(bag_path)], default_typestore=TYPESTORE) as reader:
        print('Reading image messages')
        # We are reading the messages in topic pairs to match color and depth
        for i in range(len(args.color_topics)):
            read_message = 0
            open_color_images = []
            open_depth_images = []
            connections = [
                x for x in reader.connections if x.topic in [args.color_topics[i],
                                                             args.depth_topics[i]]]
            matched_images = 0
            print(f'Reading images of sensor {i+1} of {len(args.color_topics)}')
            for connection, _, rawdata in reader.messages(connections=connections):
                msg = TYPESTORE.deserialize_cdr(rawdata, connection.msgtype)
                sensor_id = topic_to_id[connection.topic]
                integrated = False
                if connection.topic in args.color_topics:
                    for idx, depth_image in enumerate(open_depth_images):
                        depth_time = Time(seconds=depth_image.header.stamp.sec,
                                          nanoseconds=depth_image.header.stamp.nanosec)
                        color_time = Time(seconds=msg.header.stamp.sec,
                                          nanoseconds=msg.header.stamp.nanosec)
                        if abs((depth_time - color_time).nanoseconds)/1e9 < MAX_TIME_DELTA:
                            integrate(msg, depth_image, sensor_id, tf_buffer, inspection)
                            integrated = True
                            matched_images += 1
                            open_depth_images.pop(idx)
                            break
                    if not integrated:
                        open_color_images.append(msg)
                else:
                    for idx, color_image in enumerate(open_color_images):
                        depth_time = Time(seconds=msg.header.stamp.sec,
                                          nanoseconds=msg.header.stamp.nanosec)
                        color_time = Time(seconds=color_image.header.stamp.sec,
                                          nanoseconds=color_image.header.stamp.nanosec)
                        if abs((depth_time - color_time).nanoseconds)/1e9 < MAX_TIME_DELTA:
                            integrate(color_image, msg, sensor_id, tf_buffer, inspection)
                            integrated = True
                            matched_images += 1
                            open_color_images.pop(idx)
                            break
                    if not integrated:
                        open_depth_images.append(msg)

                # Print progress feedback for the user
                read_message += 1
                if (read_message) % 100 == 0:
                    percentage_done = (read_message) / \
                        (topic_message_numbers[args.color_topics[i]]
                         + topic_message_numbers[args.depth_topics[i]]) * 100
                    print(f'\rProcessing... {percentage_done: .2f}% done', end='')
                    if percentage_done > args.read_percentage:
                        break
            print('\rProcessing... 100.00% done')
            print(f'{matched_images} images were matched correctly.')
            print(f'The following count of images were not matchable for sensor {i}\n  color: \
                  {len(open_color_images)}\n  depth: {len(open_depth_images)}')


def process_bag(bag_path, args):
    """Create a Vinspect object and read the neccessary topics in the bag."""
    # Create a Vinspect object
    # TODO sensor types need to be made configurable or set automatically based on slected topics
    inspection = Inspection(['RGBD', 'RGBD', 'RGBD'],
                            inspection_space_3d_min=args.inspection_space_min,
                            inspection_space_3d_max=args.inspection_space_max)
    inspection.reinitialize_TSDF(args.voxel_length, args.sdf_trunc)
    num_cameras = len(args.color_topics)
    if len(args.color_topics) != len(args.depth_topics):
        print('Image and depth topics do not match')
        exit()

    topic_to_id = {}
    for i in range(num_cameras):
        topic_to_id[args.color_topics[i]] = i
        topic_to_id[args.depth_topics[i]] = i

    try:
        with AnyReader([Path(bag_path)], default_typestore=TYPESTORE) as reader:
            # Check if all topics are present
            for topic in args.color_topics + args.depth_topics + args.info_topics + [TF_TOPIC] + \
                    [TF_STATIC_TOPIC]:
                exists_in_bag = False
                for connection in reader.connections:
                    if topic == connection.topic:
                        exists_in_bag = True
                        break
                if not exists_in_bag:
                    print(f'Could not find requested topic {topic} in bag. \
                           Following topics are in the bag:')
                    for connection in reader.connections:
                        print(connection.topic, connection.msgtype)
                    exit()

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
                if topic_message_numbers[camera_topic] == 0:
                    print(f'Did not find camera info messages for all cameras! Problem with \
                          camera_info topic {camera_topic}')
                print('Currently ignored!!!!!!')
    except UnicodeDecodeError:
        print('Error decoding bag file. Are you sure that you provided the path to the folder and \
              not to the mcap file?')

    tf_buffer = read_tf(bag_path, topic_message_numbers, args)
    read_camera_infos(bag_path, inspection, args)
    read_images(bag_path, inspection, topic_to_id, topic_message_numbers, tf_buffer, args)
    inspection.finish()

    # Provide statistics to the user when finished reading the bag
    print('Statistics:')
    print(f'Integrated images {inspection.get_integrated_images_count()}')
    if inspection.get_integrated_images_count() == 0:
        print('No mesh could be reconstructed. Please check if the inspection space boundaries \
              are correct.')
    else:
        mesh_path = '/tmp/extracted_mesh.ply'
        # save the mesh
        inspection.save_dense_reconstruction(mesh_path)
        print(f'Mesh saved to {mesh_path}')
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process a ROS2 bag file.')
    parser.add_argument('bag_path', help='Path to the ROS2 bag file.')
    parser.add_argument('--color-topics', type=str, nargs='+',
                        help='The color topics that should be used')
    parser.add_argument('--depth-topics', type=str, nargs='+',
                        help='The depth topics that should be used, in matching order to the color \
                            topics')
    parser.add_argument('--info-topics', type=str, nargs='+',
                        help='The camera info topics that should be used, in matching order to the \
                            color topics')

    parser.add_argument('--voxel-length', type=float, default=0.005, help='Voxel length [m]')
    parser.add_argument('--sdf-trunc', type=float, default=0.05,
                        help='SDF truncation distance [m]')
    parser.add_argument('--inspection-space-min', type=float, nargs='+', default=[-1.0, -1.0, -1.0],
                        help='Inspection space minimal boundaries as vector x y z in m')
    parser.add_argument('--inspection-space-max', type=float, nargs='+', default=[1.0, 1.0, 1.0],
                        help='Inspection space maximal boundaries as vector x y z in m')
    parser.add_argument('--read-percentage', type=float, default=100,
                        help='How much of the rosbag should be read. Useful for quick testing.')
    parser.add_argument('--apply-tf-hack', action='store_true', default=False,
                        help='Just a quick hack for a wrong tf')
    args = parser.parse_args()
    process_bag(args.bag_path, args)
