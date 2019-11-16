'''
Authors: Alex Buck, Thomas Huang, Arjun Sree Manoj

Data Acquired using realsense-recorder.py provided by Open3D

Post-Processing Color and Depth Files:
1) Stream Camera to obtain camera instrinsics
2) Create and register RGBD Images
3) Convert to Pointclouds and register them
4) Account for errors in registration and remove them
5) Use registered PointCloud to reconstruct polygon using Ball Pivoting Algorithm
6) Export reconstructed object

'''

# Library to track time of each process and compute frames per second
from datetime import datetime

# Intel Realsense SDK
import pyrealsense2 as rs

import numpy as np

# OpenCV Library to compute baseline matching
import cv2

# Import Open3D opencv_pose_estimation python file to compute baseline matching for non-adjacent RGBD Images
from src import opencv_pose_estimation

# Open3D Library - Modern 3D Processing Library
import open3d as o3d

# Library to manage large memory allocations
import copy

# Use keys to control flow of post-processing methods
import keyboard

from os import listdir, makedirs
from os.path import exists, isfile, join, splitext
import re


# Sorts files in numerical order
def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)


# Obtain files from provided path and sort them
def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [
            path + f
            for f in listdir(path)
            if isfile(join(path, f)) and splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list


# join pathname to all folders present under the umbrella folder
def add_if_exists(path_dataset, folder_names):
    for folder_name in folder_names:
        if exists(join(path_dataset, folder_name)):
            path = join(path_dataset, folder_name)
    return path


# obtain color and depth folders
def get_rgbd_folders(path_dataset):
    path_color = add_if_exists(path_dataset, ["image/", "rgb/", "color/"])
    path_depth = join(path_dataset, "depth/")
    return path_color, path_depth


# obtain color and depth files from folders
def get_rgbd_file_lists(path_dataset):
    path_color, path_depth = get_rgbd_folders(path_dataset)
    color_files = get_file_list(path_color, ".jpg") + \
                  get_file_list(path_color, ".png")
    depth_files = get_file_list(path_depth, ".png")
    return color_files, depth_files


# Obtain instrinsics of the camera
def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)

    return out


def create_one_RGBD(color_file, depth_file):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_file, depth_file, depth_scale=1.0/depth_scale,
                                                                    depth_trunc=clipping_distance_in_meters,
                                                                    convert_rgb_to_intensity=False)
    return rgbd_image


def register_adjacent_RGBDs(s,t, color_files, depth_files, intrinsics):

    source_rgbd = create_one_RGBD(color_files[s], depth_files[s])
    target_rgbd = create_one_RGBD(color_files[t], depth_files[t])

    option = o3d.odometry.OdometryOption()
    odometry_init = np.identity(4)

    [success, transformation, info] = o3d.odometry.compute_rgbd_odometry(source_rgbd, target_rgbd, intrinsics, odometry_init,
                                                                         o3d.odometry.RGBDOdometryJacobianFromHybridTerm(),
                                                                         option)

    return success, transformation, info


def register_non_adjacent_RGBDs(s, t, color_files, depth_files, intrinsics):

    source_rgbd = create_one_RGBD(color_files[s], depth_files[s])
    target_rgbd = create_one_RGBD(color_files[t], depth_files[t])

    option = o3d.odometry.OdometryOption()

    # Compute wide baseline matching for non-adjacent images (align them by matching sparse features)
    success_matching, odometry_init = opencv_pose_estimation.pose_estimation(source_rgbd, target_rgbd, intrinsics, False)

    if success_matching:

        [success, transformation, info] = o3d.odometry.compute_rgbd_odometry(source_rgbd, target_rgbd, intrinsics,
                                                                             odometry_init,
                                                                             o3d.odometry.RGBDOdometryJacobianFromHybridTerm(),
                                                                             option)
    return success, transformation, info


def create_RGBD_fragments (s, t, fragment_size, color_files, depth_files, intrinsics):
    pose_graph = o3d.registration.PoseGraph()
    transformation = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(transformation))
    for s in range(len(color_files)):
        for t in range(s+1, len(color_files)):
            if t == s + 1:
                [success, trans, info] = register_adjacent_RGBDs(s, t, color_files, depth_files,intrinsics)
                transformation = np.dot(trans, transformation)
                trans_inv = np.linalg.inv(transformation)
                pose_graph.nodes.append(o3d.registration.PoseGraphNode(trans_inv))
                pose_graph.edges.append(o3d.registration.PoseGraphEdge(s, t, trans, info, uncertain=False))
            elif (s % fragment_size == 0) and (t % fragment_size == 0):
                [success, trans, info] = register_non_adjacent_RGBDs(s, t, color_files, depth_files, intrinsics)
                if success:
                    pose_graph.edges.append(o3d.registration.PoseGraphEdge(s, t, trans, info, uncertain=True))
    return pose_graph

# TODO: optimize created pose_graph for tighter alignment
def optimize_pose_for_frag():
    return True

# TODO: Helper function to run optimization of RGBD fragment
def pose_optimization():
    return True

# TODO: Integrate RGBD images into a mesh that can be projected onto 3D surface
def integrate_RGBD():
    return True

# TODO: Convert RGBD fragment into pointcloud fragment for registration
def RGBD_to_pointcloud():
    return True

'''
Variables Used:
intrinsics_obtained: boolean to check if intrinsics of the camera was retrieved
s: counter for objects used as 'source'
t: counter for objects used as 'target'
'''
intrinsics_obtained = False
s = 0
t = 0
''''''

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Start streaming
profile = pipeline.start(config)

# Config properties of the IntelRealSense D435
depth_sensor = profile.get_device().first_depth_sensor()
laserpwr = depth_sensor.get_option(rs.option.laser_power)
depth_sensor.set_option(rs.option.emitter_enabled, 1)
depth_sensor.set_option(rs.option.laser_power, laserpwr)
depth_sensor.set_option(rs.option.depth_units, 0.0001)
depth_sensor.set_option(rs.option.visual_preset, 3)
depth_scale = depth_sensor.get_depth_scale()

# Truncate how far sensor can see
clipping_distance_in_meters = 0.75
clipping_distance = clipping_distance_in_meters / depth_scale

# Aligning depth frame with color frame
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:

    while True:

        dt0 = datetime.now()

        # Wait for the next set of frames from the camera
        frames = pipeline.wait_for_frames()

        # Align Depth and Color Frame
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Try This:
        intrinsic = get_intrinsic_matrix(color_frame)
        intrinsics_obtained = True

        #Obtain intrinsics and switch to post-process methods
        if intrinsics_obtained:
            break

        # Make sure frames come in together
        if not aligned_depth_frame or not color_frame:  # or not depth_2 or not color_2:
            continue

        process_time = datetime.now() - dt0
        print("FPS: " + str(1 / process_time.total_seconds()))

finally:
    pipeline.stop()