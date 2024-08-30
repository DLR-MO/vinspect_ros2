import faulthandler
from rosbags.rosbag2 import Reader
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import yaml
import rclpy
import rclpy.time
from vinspect.vinspect_py import Inspection
from geometry_msgs.msg import TransformStamped
from pathlib import Path
from tf2_ros import Buffer
from rclpy.duration import Duration
from geometry_msgs.msg import TransformStamped
from pprint import pprint
import open3d as o3d
from rclpy.time import Time
import numpy as np
from transforms3d.affines import compose
from transforms3d.quaternions import quat2mat
from rosbags.image import message_to_cvimage

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


def deserialize_to_msg(desirialized_msg):
    # the rosbag library does not provide the real message object, so we need to create it manually
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
                buffer.set_transform_static(deserialize_to_msg(
                    deserizled_transform), 'rosbag_reader')
            read_message += 1
            if (read_message) % 1 == 0:
                print(f"\rProcessing... {
                    read_message / topic_message_numbers[TF_STATIC_TOPIC] * 100:.2f}% done", end='')

        read_message = 0
        print('\nReading dynamic TF messages')
        connections = [x for x in reader.connections if x.topic == TF_TOPIC]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = TYPESTORE.deserialize_cdr(rawdata, connection.msgtype)
            # each message can contain multiple transforms
            for deserizled_transform in msg.transforms:
                buffer.set_transform(deserialize_to_msg(deserizled_transform), 'rosbag_reader')
            read_message += 1
            if (read_message) % 1000 == 0:
                percentage_done = read_message / topic_message_numbers[TF_TOPIC] * 100
                print(f"\rProcessing... {percentage_done: .2f}% done", end='')
                if percentage_done > args.read_percentage:
                    break
        print(f"\rProcessing... 100.00% done")
    return buffer


def read_camera_infos(bag_path, inspection, args):
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


def integrate(des_color, des_depth, id, buffer, inspection):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.cpu.pybind.geometry.Image(message_to_cvimage(des_color)), o3d.cpu.pybind.geometry.Image(
        message_to_cvimage(des_depth)), depth_scale=DEPTH_SCALE, depth_trunc=DEPTH_TRUNC, convert_rgb_to_intensity=False)
    # get corresponding pose from tf2
    try:
        # TODO check if it makes sense that we use the color msg as frame of reference
        trans = buffer.lookup_transform(des_color.header.frame_id, WORLD_LINK, Time(
            seconds=des_color.header.stamp.sec, nanoseconds=des_color.header.stamp.nanosec) - Time(nanoseconds=TIME_OFFSET))
    except Exception as e:
        print(e)
        print('Ignored image that could not be transformed')
        return
    # pprint(trans)
    t = trans.transform
    affine_matrix = compose(np.array([t.translation.x, t.translation.y, t.translation.z]), quat2mat(
        np.array([t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z])), np.array([1.0, 1.0, 1.0]))
    # pprint(affine_matrix)
    # print('-----------')
    inspection.integrate_image(rgbd_image, id, affine_matrix)


def read_images(bag_path, inspection, topic_to_id, topic_message_numbers, tf_buffer, args):
    with AnyReader([Path(bag_path)], default_typestore=TYPESTORE) as reader:
        # now we can go through the images and integrate them
        # we need to always find the matching pair of color and depth image
        print('Reading image messages')
        # We are reading the messages in pairs to match color and depth
        for i in range(len(args.color_topics)):
            read_message = 0
            open_color_images = []
            open_depth_images = []
            connections = [
                x for x in reader.connections if x.topic in [args.color_topics[i], args.depth_topics[i]]]
            matched_images = 0
            print(f'Reading images of sensor {i+1} of {len(args.color_topics)}')
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = TYPESTORE.deserialize_cdr(rawdata, connection.msgtype)
                id = topic_to_id[connection.topic]
                integrated = False
                if connection.topic in args.color_topics:
                    for idx, depth_image in enumerate(open_depth_images):
                        depth_time = Time(seconds=depth_image.header.stamp.sec,
                                          nanoseconds=depth_image.header.stamp.nanosec)
                        color_time = Time(seconds=msg.header.stamp.sec,
                                          nanoseconds=msg.header.stamp.nanosec)
                        if abs((depth_time - color_time).nanoseconds)/1e9 < MAX_TIME_DELTA:
                            integrate(msg, depth_image, id, tf_buffer, inspection)
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
                            integrate(color_image, msg, id, tf_buffer, inspection)
                            integrated = True
                            matched_images += 1
                            open_color_images.pop(idx)
                            break
                    if not integrated:
                        open_depth_images.append(msg)

                # Print progress feedback for the user
                read_message += 1
                if (read_message) % 100 == 0:
                    percentage_done = (
                        read_message) / (topic_message_numbers[args.color_topics[i]] + topic_message_numbers[args.depth_topics[i]]) * 100
                    print(f"\rProcessing... {percentage_done: .2f}% done", end='')
                    if percentage_done > args.read_percentage:
                        break
                    # o3d.visualization.draw_geometries([inspection.extract_dense_reconstruction()])
            print(f"\rProcessing... 100.00% done")
            print(f'{matched_images} images were matched correctly.')
            print(f'The following count of images were not matchable for sensor {i}\n  color: {
                len(open_color_images)}\n  depth: {len(open_depth_images)}')


def process_bag(bag_path, args):
    # Create a Vinspect object
    # TODO the sensor types need to be made configurable
    inspection = Inspection(["RGBD", "RGBD", "RGBD"], inspection_space_min=args.inspection_space_min,
                            inspection_space_max=args.inspection_space_max)
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
            for topic in args.color_topics + args.depth_topics + args.info_topics + [TF_TOPIC] + [TF_STATIC_TOPIC]:
                exists_in_bag = False
                for connection in reader.connections:
                    if topic == connection.topic:
                        exists_in_bag = True
                        break
                if not exists_in_bag:
                    print(f'Could not find requested topic {
                          topic} in bag. Following topics are in the bag:')
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
            print(f'Using bag with messages from {reader.start_time/1e9} to {
                  reader.end_time/1e9} with a duration of {(reader.end_time-reader.start_time)/1e9} s')
            pprint(topic_message_numbers)

            # Sanity check
            for camera_topic in args.info_topics:
                if topic_message_numbers[camera_topic] == 0:
                    print(f'Did not find camera info messages for all cameras! Problem with camera_info topic {
                          camera_topic}')
                print('Currently ignored!!!!!!')
    except UnicodeDecodeError:
        print('Error decoding bag file. Are you sure that you provided the path to the folder and not to the mcap file?')

    tf_buffer = read_tf(bag_path, topic_message_numbers, args)
    read_camera_infos(bag_path, inspection, args)
    read_images(bag_path, inspection, topic_to_id, topic_message_numbers, tf_buffer, args)

    # Provide statistics to the user when finished reading the bag
    print("Statistics:")
    print(f'Integrated images {inspection.get_integrated_images_count()}')
    if inspection.get_integrated_images_count() == 0:
        print(f'No mesh could be reconstructed. Please check if the inspection space boundaries are correct.')
    mesh = inspection.extract_dense_reconstruction()
    print(mesh)
    # save the mesh
    mesh.compute_triangle_normals()
    o3d.io.write_triangle_mesh("mesh.stl", mesh)
    o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process a ROS2 bag file.")
    parser.add_argument("bag_path", help="Path to the ROS2 bag file.")
    parser.add_argument('--color-topics', type=str, nargs='+',  # '/camera/camera/color/image_rect_raw','/camera1/camera1/color/image_rect_raw',/camera2/camera2/color/image_rect_raw']
                        help='The color topics that should be used')
    parser.add_argument('--depth-topics', type=str, nargs='+',  # '/camera/camera/depth/image_rect_raw', '/camera1/camera1/depth/image_rect_raw', '/camera2/camera2/depth/image_rect_raw'
                        help='The depth topics that should be used, in matching order to the color topics')
    parser.add_argument('--info-topics', type=str, nargs='+',  # '/camera/camera/color/camera_info',  '/camera1/camera1/color/camera_info','/camera2/camera2/color/camera_info']
                        help='The camera info topics that should be used, in matching order to the color topics')

    parser.add_argument('--voxel-length', type=float, default=0.005, help=f'Voxel length')
    parser.add_argument('--sdf-trunc', type=int, default=0.50, help=f'SDF truncation distance')
    parser.add_argument('--inspection-space-min', type=float, nargs='+', default=[-1.0, -1.0, -1.0],  # [-0.85,0.5,0.0]#m
                        help=f'Inspection space minimal boundaries as vector x y z in m')
    parser.add_argument('--inspection-space-max', type=float, nargs='+', default=[1.0, 1.0, 1.0],  # [-0.25,1.0,0.3]#m
                        help=f'Inspection space maximal boundaries as vector x y z in m')
    parser.add_argument('--read-percentage', type=float, default=100,
                        help=f'How much of the rosbag should be read. Useful for quick testing.')
    args = parser.parse_args()
    process_bag(args.bag_path, args)
