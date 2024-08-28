import os
import cv2
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from vinspect.vinspect_py import Inspection
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from pathlib import Path
from tf2_ros import Buffer
from rclpy.duration import Duration
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion, Transform
from pprint import pprint
import open3d as o3d
from rclpy.time import Time
import numpy as np
from transforms3d.affines import compose
from transforms3d.quaternions import quat2mat


IMAGE_TOPICS = ['/camera/camera/color/image_rect_raw', '/camera2/camera2/color/image_rect_raw']
DEPTH_TOPICS = ['/camera/camera/depth/image_rect_raw', '/camera2/camera2/depth/image_rect_raw']
CAMERA_TOPICS = ['/camera/camera/color/camera_info', '/camera2/camera2/color/camera_info']
TF_TOPIC = '/tf'
TF_STATIC_TOPIC = '/tf_static'
ALL_TOPICS = IMAGE_TOPICS + DEPTH_TOPICS + CAMERA_TOPICS + [TF_TOPIC] + [TF_STATIC_TOPIC]

DEPTH_SCALE = 1000.0
DEPTH_TRUNC = 0.50  # m
WORLD_LINK = 'world'
VOXEL_LENGTH = 0.005  # m
SDF_TRUNC = 0.50  # m
INSPECTION_SPACE_MIN = [-1.0, -1.0, -1.0]  # [-0.85,0.5,0.0]#m
INSPECTION_SPACE_MAX = [1.0, 1.0, 1.0]  # [-0.25,1.0,0.3]#m
TIME_OFFSET = 0.0  # 0.99*1e9 #ns


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


def process_bag(bag_path):
    # Create a VInspect object
    inspection = Inspection(["RGBD"], inspection_space_min=INSPECTION_SPACE_MIN,
                            inspection_space_max=INSPECTION_SPACE_MAX)
    inspection.reinitialize_TSDF(VOXEL_LENGTH, SDF_TRUNC)
    read_message = 0
    num_cameras = len(IMAGE_TOPICS)
    if len(IMAGE_TOPICS) != len(DEPTH_TOPICS):
        print('Image and depth topics do not match')
        exit()

    topic_to_id = {}
    for i in range(num_cameras):
        topic_to_id[IMAGE_TOPICS[i]] = i
        topic_to_id[DEPTH_TOPICS[i]] = i

    # Read the bag in chronological order
    typestore = get_typestore(Stores.LATEST)
    try:
        with Reader(bag_path) as reader:
            # Check if all topics are present
            for topic in ALL_TOPICS:
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
            for topic_data in reader.metadata['topics_with_message_count']:
                topic_message_numbers[topic_data['topic_metadata']
                                      ['name']] = topic_data['message_count']
            print(f'Using bag with messages from {reader.start_time/1e9} to {
                  reader.end_time/1e9} with a duration of {(reader.end_time-reader.start_time)/1e9} s')
            pprint(topic_message_numbers)
            all_image_count = 0
            for topic in IMAGE_TOPICS + DEPTH_TOPICS:
                all_image_count += topic_message_numbers[topic]

            # Sanity check
            for camera_topic in CAMERA_TOPICS:
                if topic_message_numbers[camera_topic] == 0:
                    print(f'Did not find camera info messages for all cameras! Problem with camera_info topic {
                          camera_topic}')
                print('Currently ignored!!!!!!')

            # read camera infos one by one to make sure we match sensor IDs correctly
            sensor_id = 0
            for camera_topic in CAMERA_TOPICS:
                connections = [x for x in reader.connections if x.topic == camera_topic]
                for connection, timestamp, rawdata in reader.messages(connections=connections):
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    intrinsic = o3d.camera.PinholeCameraIntrinsic(
                        msg.width, msg.height, msg.k[0], msg.k[4], msg.k[2], msg.k[5])
                    inspection.set_intrinsic(intrinsic, sensor_id)
                sensor_id += 1

            # First read all tf and tf_static messages into the buffer
            # Create tf2 buffer with a buffer length of the whole rosbag
            length_ns = reader.duration
            buffer = Buffer(Duration(seconds=length_ns/1e9, nanoseconds=length_ns % 1e9))
            # Fill with data
            print('Reading static TF messages')
            connections = [x for x in reader.connections if x.topic == TF_STATIC_TOPIC]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
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
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                # each message can contain multiple transforms
                for deserizled_transform in msg.transforms:
                    buffer.set_transform(deserialize_to_msg(deserizled_transform), 'rosbag_reader')
                read_message += 1
                if (read_message) % 100 == 0:
                    print(f"\rProcessing... {
                          read_message / topic_message_numbers[TF_TOPIC] * 100:.2f}% done", end='')
            print(f"\rProcessing... 100.00% done")

            def integrate(des_color, des_depth, id):
                color_array = np.frombuffer(des_color.data, dtype=np.uint8)
                color_reshaped_array = color_array.reshape(
                    des_color.height, des_color.width, 3).astype(np.uint8)
                # color_img = o3d.cpu.pybind.geometry.Image(color_reshaped_array)

                depth_array = np.frombuffer(des_depth.data, dtype=np.uint16)
                reshaped_array = depth_array.reshape(
                    des_depth.height, des_depth.width).astype(np.uint16)
                # depth_img = o3d.cpu.pybind.geometry.Image(reshaped_array)
                # o3d.visualization.draw_geometries([color_img])
                # o3d.visualization.draw_geometries([depth_img])
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.cpu.pybind.geometry.Image(color_reshaped_array), o3d.cpu.pybind.geometry.Image(
                    reshaped_array), depth_scale=DEPTH_SCALE, depth_trunc=DEPTH_TRUNC, convert_rgb_to_intensity=False)
                # o3d.visualization.draw_geometries([rgbd_image])
                # get corresponding pose from tf2
                try:
                    trans = buffer.lookup_transform(msg.header.frame_id, WORLD_LINK, Time(
                        seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec) - Time(nanoseconds=TIME_OFFSET))
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

            # now we can go through the images and integrate them
            # we need to always find the matching pair of color and depth image
            print('Reading image messages')
            read_message = 0
            open_color_images = [[]]*num_cameras
            open_depth_images = [[]]*num_cameras
            connections = [x for x in reader.connections if x.topic in IMAGE_TOPICS + DEPTH_TOPICS]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                id = topic_to_id[connection.topic]
                integrated = False
                if connection.topic in IMAGE_TOPICS:
                    for depth_image in open_depth_images[id]:
                        if depth_image.header.stamp.sec == msg.header.stamp.sec and depth_image.header.stamp.nanosec == msg.header.stamp.nanosec:
                            integrate(msg, depth_image, id)
                            integrated = True
                            break
                    if not integrated:
                        open_color_images[id].append(msg)
                else:
                    for color_image in open_color_images[id]:
                        if color_image.header.stamp.sec == msg.header.stamp.sec and color_image.header.stamp.nanosec == msg.header.stamp.nanosec:
                            integrate(color_image, msg, id)
                            integrated = True
                            break
                    if not integrated:
                        open_depth_images[id].append(msg)

                # Print progress feedback for the user
                read_message += 1
                if (read_message) % 100 == 0:
                    print(f"\rProcessing... {(read_message) /
                          all_image_count * 100:.2f}% done", end='')
                    # o3d.visualization.draw_geometries([inspection.extract_dense_reconstruction()])
            print(f"\rProcessing... 100.00% done")
            print(f'The following count of images were not matchable\n  color: {
                  len(open_color_images)}\n  depth: {len(open_depth_images)}')
    except UnicodeDecodeError:
        print('Error decoding bag file. Are you sure that you provided the path to the folder and not to the mcap file?')
    # Provide statistics to the user when finished reading the bag
    print("Statistics:")
    print(f"Processed images: {read_message}")
    print(f'Integrated images {inspection.get_integrated_images_count()}')
    mesh = inspection.extract_dense_reconstruction()
    print(mesh)
    o3d.visualization.draw_geometries([mesh])
    # print(f"Mean processing time per image: {inspection.mean_process_time:.4f} s")
    # print(f"Total processing time: {inspection.total_process_time:.4f} s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process a ROS2 bag file.")
    parser.add_argument("bag_path", help="Path to the ROS2 bag file.")
    args = parser.parse_args()
    process_bag(args.bag_path)
