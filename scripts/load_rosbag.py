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

IMAGE_TOPICS = ['/camera/camera/color/image_rect_raw']
DEPTH_TOPICS = ['/camera/camera/depth/image_rect_raw']
CAMERA_TOPICS = ['/camera/camera/color/camera_info']
TF_TOPIC = '/tf'
TF_STATIC_TOPIC = '/tf_static'

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
    vi = Inspection(["RGBD"], inspection_space_min=[-1.0,-1.0,-1.0], inspection_space_max=[1.0,1.0,1.0])
    read_message = 0

    # Read the bag in chronological order
    typestore = get_typestore(Stores.LATEST)    
    try:
        with Reader(bag_path) as reader:
            # Get number of message on the used topics for progressbar later
            topic_message_numbers = {}
            for topic_data in reader.metadata['topics_with_message_count']:
                topic_message_numbers[topic_data['topic_metadata']['name']] = topic_data['message_count']                        
            # maybe call this for --help
            #for connection in reader.connections:
            #    print(connection.topic, connection.msgtype)
            # First read all tf and tf_static messages into the buffer
            # Create tf2 buffer with a buffer length of the whole rosbag
            length_ns = reader.duration
            buffer = Buffer(Duration(seconds=length_ns/10e6, nanoseconds=length_ns%10e6))            
            # Fill with data
            print('Reading static TF messages')
            connections = [x for x in reader.connections if x.topic == TF_STATIC_TOPIC]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                for deserizled_transform in msg.transforms:
                    buffer.set_transform_static(deserialize_to_msg(deserizled_transform), 'rosbag_reader')
                read_message+=1
                if (read_message) % 1 == 0:
                    print(f"\rProcessing... {read_message / topic_message_numbers[TF_STATIC_TOPIC] * 100:.2f}% done", end='')
            
            read_message = 0
            print('\nReading dynamic TF messages')
            connections = [x for x in reader.connections if x.topic == TF_TOPIC]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                # each message can contain multiple transforms                
                for deserizled_transform in msg.transforms:                    
                    buffer.set_transform(deserialize_to_msg(deserizled_transform), 'rosbag_reader')
                read_message+=1
                if (read_message) % 100 == 0:
                    print(f"\rProcessing... {read_message/ topic_message_numbers[TF_TOPIC] * 100:.2f}% done", end='')
            
            # read 

            if False:
                # Print connections that we use
                connections = [x for x in reader.connections if x.topic in TOPIC_NAMES]
                for connection in connections:
                    print(connection.topic, connection.msgtype)
                
                for connection, timestamp, rawdata in reader.messages(connections=connections):
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                                    
                    #todo somehow use the tf messages to make it possible to get transformations. Maybe run a robot_state publisher node in here                

                    # Check if image message is present                
                    #topic_name = topic.topic.split('/')[-1]                
                    # Call the integrateImage method of the inspection object
                    #vi.integrateImage(img_msg, depth_msg, cam_info_msg)
                    try:
                        print(f"\r{msg.header.timestamp}", end='')
                    except:
                        pass

            # Print progress feedback for the user
            read_message+=1
            if (read_message) % 100 == 0:
                print(f"\rProcessing... {(read_message) / reader.message_count * 100:.2f}% done", end='')
    except UnicodeDecodeError:
        print('Error decoding bag file. Are you sure that you provided the path to the folder and not to the mcap file?')
    # Provide statistics to the user when finished reading the bag
    print("Statistics:")
    print(f"Processed images: {len(vi.images)}")
    print(f"Mean processing time per image: {vi.mean_process_time:.4f} s")
    print(f"Total processing time: {vi.total_process_time:.4f} s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process a ROS2 bag file.")
    parser.add_argument("bag_path", help="Path to the ROS2 bag file.")
    args = parser.parse_args()
    process_bag(args.bag_path)
