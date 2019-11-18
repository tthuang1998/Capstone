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


# creates RGBD image from color image and depth image
def create_one_RGBD(color_file, depth_file):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_file, depth_file, depth_scale=1.0/depth_scale,
                                                                    depth_trunc=clipping_distance_in_meters,
                                                                    convert_rgb_to_intensity=False)
    return rgbd_image

# register locally RGBD Images using RGBD Odometry
def register_adjacent_RGBDs(s,t, color_files, depth_files, intrinsics):

    source_rgbd = create_one_RGBD(color_files[s], depth_files[s])
    target_rgbd = create_one_RGBD(color_files[t], depth_files[t])

    option = o3d.odometry.OdometryOption()
    odometry_init = np.identity(4)

    [success, transformation, info] = o3d.odometry.compute_rgbd_odometry(source_rgbd, target_rgbd, intrinsics, odometry_init,
                                                                         o3d.odometry.RGBDOdometryJacobianFromHybridTerm(),
                                                                         option)

    return success, transformation, info

# register globally using OpenCV ORB feature to find sparse overlap
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
    else:
        odometry_init = np.identity(4)
        [success, transformation, info] = o3d.odometry.compute_rgbd_odometry(source_rgbd, target_rgbd, intrinsics,
                                                                             odometry_init,
                                                                             o3d.odometry.RGBDOdometryJacobianFromHybridTerm(),
                                                                             option)

    return success, transformation, info

'''
Create pose graph for all created RGBD images and then register globally every frame
'''
def create_RGBD_posegraph(fragment_size, color_files, depth_files, intrinsics):
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


# optimize posegraph based on specific optimization parameters
def optimize_pose(pose_graph, max_correspondence_dist, loop_closure, edge_prune, ref_node):
    method = o3d.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.registration.GlobalOptimizationConvergenceCriteria()
    option = o3d.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_dist,
        edge_prune_threshold=edge_prune,
        preference_loop_closure=loop_closure,
        reference_node=ref_node)
    o3d.registration.global_optimization(pose_graph, method, criteria, option)
    return pose_graph

# combine RGBD images into a TSDF Volume and create the mesh
def integrate_RGBD(pose_graph, color_files, depth_files, voxel_cube_size):
    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length= voxel_cube_size/ 512.0,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)
    for i in range(len(pose_graph.nodes)):
        rgbd = create_one_RGBD(color_file=color_array[i], depth_file=depth_array[i])
        pose = pose_graph.nodes[i].pose
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh

# convert RGBD Image into pointcloud
def RGBD_to_pointcloud(rgbd, intrinsics):
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.geometry.PointCloud.estimate_normals(pc, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pc


# TODO: registration -> pairwise registration for adjacent and global registration to global pointcloud
'''
Steps needed:
1) Create a global pointcloud that stores registered clouds
2) Create a pose graph
3) register adjacent frames using pairwise
4) register the adjacent-frame-pointcloud to the global pointcloud
4) if registration done correctly, optimize pose graph
5) transform target pointcloud based on pose graph
6) output global pointcloud
'''

def pairwise_registration(source, target, intrinsics, voxel_radius):

    source_pc = RGBD_to_pointcloud(source, intrinsics)
    target_pc = RGBD_to_pointcloud(target, intrinsics)
    current_transformation = np.identity(4)

    icp = o3d.registration.registration_icp(source_pc, target_pc, voxel_radius * 1.5, current_transformation,
        o3d.registration.TransformationEstimationPointToPlane())

    current_transformation = icp.transformation

    icp_fine = o3d.registration.registration_icp(source_pc, target_pc, voxel_radius, current_transformation,
        o3d.registration.TransformationEstimationPointToPlane())

    current_transformation = icp_fine.transformation

    information_icp = o3d.registration.get_information_matrix_from_point_clouds(source_pc, target_pc, voxel_radius * 1.5,
        current_transformation)

    return current_transformation, information_icp

def global_registration(source, target, intrinsics, voxel_radius):

    source_pc = RGBD_to_pointcloud(source, intrinsics)
    target_pc = RGBD_to_pointcloud(target, intrinsics)
    current_transformation = np.identity(4)

    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        source_pc, target_pc, voxel_radius,
        current_transformation)

    if current_transformation.trace() == 4.0:
        current_transformation = np.identity(4)
        information_icp = np.zeros((6, 6))

    elif information_icp[5, 5] / min(len(copy.deepcopy(source_pc.points)),
                                     len(copy.deepcopy(target_pc.points))) < 0.3:
        current_transformation = np.identity(4)
        information_icp = np.zeros((6, 6))

    return current_transformation, information_icp

def registration(rgbd_array, intrinsics, pc_list):

    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    odometry_global = np.identity(4)

    for s in range(len(rgbd_array)):
        pc_list.append(RGBD_to_pointcloud(rgbd_array[s], intrinsics))
        for t in range(s+1, len(rgbd_array)):
            if t == s + 1:
                c_odometry, information = pairwise_registration(rgbd_array[s], rgbd_array[t], intrinsics, 0.005)

                odometry = np.dot(c_odometry, odometry)
                pose_graph.nodes.append(o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(o3d.registration.PoseGraphEdge(s, t, odometry, information, uncertain=False))
            elif (s % 50 == 0) and (t % 50 == 0):
                c_odometry, information = global_registration(rgbd_array[s], rgbd_array[t], intrinsics, 0.005)
                pose_graph.edges.append(o3d.registration.PoseGraphEdge(s, t, c_odometry, information, uncertain=True))

    return pose_graph, pc_list

'''
def step_registration(source, target):

    # Get pointcloud and features for global registration
    source_temp, feature_source = setup_cloud(source)
    target_temp, feature_target = setup_cloud(target)

    # Get normal of plane for each pointcloud
    source_dir = get_normals(source_temp)
    target_dir = get_normals(target_temp)

    #Normals towards camera
    o3d.orient_normals_towards_camera_location(source_temp)
    o3d.orient_normals_towards_camera_location(target_temp)

    # Get angle between normals of pointclouds
    angle = cal_angle(source_dir, target_dir)

    if (angle < 0.05):
        rotation_dir = target_dir
    else:
        rotation_dir = source_dir

    # Registration parameter
    distance_threshold = 0.1 * 1.5

    # Global Registration

    result = o3d.registration_fast_based_on_feature_matching(source_temp, target_temp, feature_source, feature_target)
    result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
        source_temp, target_temp, feature_source, feature_target, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))

    #Refine registration

    refine_result = o3d.registration.registration_icp(
        source_temp, target_temp, distance_threshold, result.transformation,
        o3d.registration.TransformationEstimationPointToPlane())

    information_icp = o3d.get_information_matrix_from_point_clouds(source_temp, target_temp, 0.01, refine_result.transformation)
    return refine_result.transformation, information_icp
'''


class VectorArrayInterface(object):
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def __array__(self, dtype=None):
        if dtype:
            return np.array([self.x, self.y, self.z], dtype=dtype)
        else:
            return np.array([self.x, self.y, self.z])

if __name__ == "__main__":
    '''
    Variables Used:
    intrinsics_obtained: boolean to check if intrinsics of the camera was retrieved
    s: counter for objects used as 'source'
    t: counter for objects used as 'target'
    color_array = color file array of o3d.geometry.Images used to create RGBD
    depth_array = depth file array of o3d.geometry.Images used to create RGBD
    rbgd_array = array of o3d.geometry.RGBDImage
    base = pointcloud that holds all registered points
    source = poincloud to register
    target = next pointcloud to register
    '''
    intrinsics_obtained = False
    s = 0
    t = 0
    color_array = []
    depth_array = []
    rgbd_array = []
    pc_list = []
    base = o3d.geometry.PointCloud()
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
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

    filepath = "C:/Users/rjsre/PycharmProjects/Trial3/src/newdata/"
    [color_files, depth_files] = get_rgbd_file_lists(filepath)
    n_files = len(color_files)
    for i in range(0, n_files):
        color_image = o3d.io.read_image(color_files[i])
        depth_image = o3d.io.read_image(depth_files[i])
        color_array.append(color_image)
        depth_array.append(depth_image)
        rgbd = create_one_RGBD(color_image, depth_image)
        rgbd_array.append(rgbd)

    pose_graph, pc_list = registration(rgbd_array, intrinsic, pc_list)
    optimize_pose(pose_graph, 0.01, 0.1, 0.25, 0)

    for node in range(len(pc_list)-1):
        pc_list[node].transform(pose_graph.nodes[node].pose)
        base += pc_list[node]
        base = o3d.geometry.PointCloud.voxel_down_sample(base, 0.005)
    o3d.visualization.draw_geometries([base])

'''
    pose_graph = create_RGBD_posegraph(5, color_files=color_array, depth_files=depth_array, intrinsics=intrinsic)
    pose_graph_optimization = optimize_pose_for_frag(pose_graph, 0.01, 0.1, 0.25, 0)
    mesh = integrate_RGBD(pose_graph_optimization, color_files, depth_files, 4.0)
    o3d.visualization.draw_geometries([mesh])
'''


