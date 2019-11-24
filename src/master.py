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


'''
Function: takes in an ordered list of files and orders them by the number in their title

Param: file_list_ordered -> takes in a list of files that are in order
Output: takes in a list of ordered files and order them by the number in their title
'''
def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)


'''
Function: looks at the path for the folder of files and put them in an ordered list

Param: path -> path to the folder of ordered files
Output: sorted list of files in alphanumerical order
'''
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

'''
Function: path to umbrella folder that contains color and depth files will rename folders in it to the same path

Param: path_dataset -> path to the folder which contains color and depth
folder_names -> name of folders in the umbrella to rename them

Output: path name
'''
def add_if_exists(path_dataset, folder_names):
    for folder_name in folder_names:
        if exists(join(path_dataset, folder_name)):
            path = join(path_dataset, folder_name)
    return path

'''
Function: get the path to the color and depth folders

Param: path_dataset -> path to the umbrella folder
Output: path to the color and depth folder
'''
def get_rgbd_folders(path_dataset):
    path_color = add_if_exists(path_dataset, ["image/", "rgb/", "color/"])
    path_depth = join(path_dataset, "depth/")
    return path_color, path_depth


'''
Function: obtain files from both the color and depth folders

Param: path_dataset -> path to the umbrella folder
Output: obtain color and depth files, ordered, respectively
'''
def get_rgbd_file_lists(path_dataset):
    path_color, path_depth = get_rgbd_folders(path_dataset)
    color_files = get_file_list(path_color, ".jpg") + \
                  get_file_list(path_color, ".png")
    depth_files = get_file_list(path_depth, ".png")
    return color_files, depth_files


'''
Function: obtain instrinsic configuration of Intel Realsense

Param: frame -> frame received when IntelRealsense is on
Output: get intrinsics matrix configuration to use in Open3D library
'''
def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)

    return out


'''
Function: create RGBD Image from color and depth file

Param: color file -> .jpg file contains color data
depth_file -> .png file contains depth data

Output: open 3D RGBD Image
'''
def create_one_RGBD(color_file, depth_file):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_file, depth_file, depth_scale=1.0/depth_scale,
                                                                    depth_trunc=clipping_distance_in_meters,
                                                                    convert_rgb_to_intensity=False)
    return rgbd_image

'''
Function: register temporally adjacent RGBD Images

Param: s -> counter for source RGBD images used
t -> counter for target RGBD images used
color_files -> list of colored files from folder
depth_files -> list of depth files from folder
intrinsics -> internal matrix configuration of camera


Output: success bool, transformation matrix between source and target, information matrix on the transformation
'''
def register_adjacent_RGBDs(s,t, color_files, depth_files, intrinsics):

    # obtain RGBD from color and depth files
    source_rgbd = create_one_RGBD(color_files[s], depth_files[s])
    target_rgbd = create_one_RGBD(color_files[t], depth_files[t])

    # setup initial transformation
    option = o3d.odometry.OdometryOption()
    odometry_init = np.identity(4)

    [success, transformation, info] = o3d.odometry.compute_rgbd_odometry(source_rgbd, target_rgbd, intrinsics, odometry_init,
                                                                         o3d.odometry.RGBDOdometryJacobianFromHybridTerm(),
                                                                         option)

    return success, transformation, info

'''
Function: register spatially non-adjacent RGBD Images

Param: s -> counter for source RGBD images used
t -> counter for target RGBD images used
color_files -> list of colored files from folder
depth_files -> list of depth files from folder
intrinsics -> internal matrix configuration of camera


Output: success bool, transformation matrix between source and target, information matrix on the transformation
'''
def register_non_adjacent_RGBDs(s, t, color_files, depth_files, intrinsics):

    # obtain RGBD from color and depth files
    source_rgbd = create_one_RGBD(color_files[s], depth_files[s])
    target_rgbd = create_one_RGBD(color_files[t], depth_files[t])

    option = o3d.odometry.OdometryOption()

    # Compute wide baseline matching for non-adjacent images (align them by matching sparse features)
    success_matching, odometry_init = opencv_pose_estimation.pose_estimation(source_rgbd, target_rgbd, intrinsics, False)

    # If frames were able to be matched, then continue with odometry transformation
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
Function: Create pose graph for all created RGBD images and then register globally every frame

Param: fragment_size -> number at which global registration will occur
color_files -> list of color images
depth_files -> list of depth images
intrinsics -> internal matrix configuration of camera

Output: pose graph for all images in dataset
'''
def create_RGBD_posegraph(fragment_size, color_files, depth_files, intrinsics):

    # Create pose graph
    pose_graph = o3d.registration.PoseGraph()

    # Set up initial transformation
    transformation = np.identity(4)

    pose_graph.nodes.append(o3d.registration.PoseGraphNode(transformation))

    # for loops that register each frame to every successor frame and appends nodes and edges to pose graph
    for s in range(len(color_files)):
        for t in range(s+1, len(color_files)):
            if t == s + 1:

                # adjacent performed for temporally adjacent frames
                [success, trans, info] = register_adjacent_RGBDs(s, t, color_files, depth_files,intrinsics)
                transformation = np.dot(trans, transformation)
                trans_inv = np.linalg.inv(transformation)

                pose_graph.nodes.append(o3d.registration.PoseGraphNode(trans_inv))
                pose_graph.edges.append(o3d.registration.PoseGraphEdge(s, t, trans, info, uncertain=False))

            # every fragment_size, register globally
            elif (s % fragment_size == 0) and (t % fragment_size == 0):

                [success, trans, info] = register_non_adjacent_RGBDs(s, t, color_files, depth_files, intrinsics)

                if success:
                    pose_graph.edges.append(o3d.registration.PoseGraphEdge(s, t, trans, info, uncertain=True))
    return pose_graph


'''
Function: Optimize Pose graph based off user configurations

Param: pose_graph -> pose graph collected from registration
max_correspondence_dist -> maximum distance between two points when performing registration
loop_closure -> preference to forcing closure with pose graph
edge_prune -> smooth out pose graph trajectory
ref_node -> starting point for pose graph

Output: optimzied pose graph
'''
def optimize_pose(pose_graph, max_correspondence_dist, loop_closure, edge_prune, ref_node):

    # LevenbergMarguardt optimization favored for Open3D registration closure
    method = o3d.registration.GlobalOptimizationLevenbergMarquardt()

    #Set configuration to optimization parameters
    criteria = o3d.registration.GlobalOptimizationConvergenceCriteria()
    option = o3d.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_dist,
        edge_prune_threshold=edge_prune,
        preference_loop_closure=loop_closure,
        reference_node=ref_node)

    #Perform the optimization
    o3d.registration.global_optimization(pose_graph, method, criteria, option)

    return pose_graph

'''
Function: Create mesh using Truncated Signed Distance Function (TSDF) volume integration from RGBD Images

Param: pose_graph -> optimized pose-graph from RGBD images
color_files -> list of color images
depth_files -> list of depth images
voxel_cube_size -> size of point to which the integration will scale to

Output: mesh of RGBD images
'''
def integrate_RGBD(pose_graph, color_files, depth_files, voxel_cube_size):

    # integrate with given pose_graph
    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length= voxel_cube_size/ 512.0,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    # Transform all RGBD images based off pose graph
    for i in range(len(pose_graph.nodes)):
        rgbd = create_one_RGBD(color_file=color_array[i], depth_file=depth_array[i])
        pose = pose_graph.nodes[i].pose
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    return mesh

'''
Function: Convert RGBD Image into open3d Pointcloud

Param: rgbd -> RGBD Image used to create pointcloud
intrinsics -> internal matrix configuration of camera

Output: pointcloud from RGBD Image
'''
def RGBD_to_pointcloud(rgbd, intrinsics):
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Estimate normals to facilitate pointcloud registration
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

'''
Function: registration of two pointclouds to each other locally

Param: source -> source RGBD image
target -> RGBD image for registration
intrinsics -> internal matrix configuration of camera
vozel_radius -> size of voxel cube that represents the individual point the algorithm taes

Output: transformation matrix to transform target to match source, information matrix on the transformation information
'''
def pairwise_registration(source, target, intrinsics, voxel_radius):

    # Set up registration by obtaining pointclouds
    source_pc = RGBD_to_pointcloud(source, intrinsics)
    target_pc = RGBD_to_pointcloud(target, intrinsics)
    current_transformation = np.identity(4)

    # Perform Iterative Closeset Point Algorithm to register both pointclouds in a coarse fashion
    icp = o3d.registration.registration_icp(source_pc, target_pc, voxel_radius * 1.5, current_transformation,
        o3d.registration.TransformationEstimationPointToPlane())

    current_transformation = icp.transformation

    # Perform Iterative Closeset Point Algorithm to register both pointclouds in a fine fashion
    icp_fine = o3d.registration.registration_icp(source_pc, target_pc, voxel_radius, current_transformation,
        o3d.registration.TransformationEstimationPointToPlane())

    current_transformation = icp_fine.transformation

    information_icp = o3d.registration.get_information_matrix_from_point_clouds(source_pc, target_pc, voxel_radius * 1.5,
        current_transformation)

    return current_transformation, information_icp


'''
Function: registration of two pointclouds in the global space

Param: source -> source RGBD image
target -> RGBD image for registration
intrinsics -> internal matrix configuration of camera
vozel_radius -> size of voxel cube that represents the individual point the algorithm taes

Output: transformation matrix to transform target to match source, information matrix on the transformation information
'''
def global_registration(source, target, intrinsics, voxel_radius):

    # Setup by creating pointclouds
    source_pc = RGBD_to_pointcloud(source, intrinsics)
    target_pc = RGBD_to_pointcloud(target, intrinsics)

    current_transformation = np.identity(4)

   # Obtain information matrix in order to justify whether these two clouds should register or not>
    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        source_pc, target_pc, voxel_radius,
        current_transformation)

    # Account for all cases in which the registration
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

def surface_reconstruction(pc, radius):
    o3d.geometry.PointCloud.estimate_normals(pc, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, radius)

    return mesh


'''
COMMENTATED CODE IN TRIAL FOR ERROR CORRECTION ALGORITHM

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


'''
Main Function to run Program for pose-processing efforts in preservation of art in #D. 
'''
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

    # Streaming loop for IntelRealSense
    try:

        while True:

            dt0 = datetime.now()

            # Wait for the next set of frames from the camera
            frames = pipeline.wait_for_frames()

            # Align Depth and Color Frame
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Get intrinsics of the camera
            intrinsic = get_intrinsic_matrix(color_frame)
            intrinsics_obtained = True

            #Obtain intrinsics and switch to post-process methods
            if intrinsics_obtained:
                break

            # Make sure frames come in together
            if not aligned_depth_frame or not color_frame:  # or not depth_2 or not color_2:
                continue

            # Obtain frames per second data
            process_time = datetime.now() - dt0
            print("FPS: " + str(1 / process_time.total_seconds()))

    finally:
        # End program once intrinsics are obtained
        pipeline.stop()

    # Path to folder that contains color and depth images
    filepath = "C:/Users/rjsre/PycharmProjects/Trial3/src/newdata/"

    [color_files, depth_files] = get_rgbd_file_lists(filepath)

    # number of files for iterative loops
    n_files = len(color_files)

    # Collect information from folders into array that hold open3D images and o3d RGBD images.
    for i in range(0, n_files):
        color_image = o3d.io.read_image(color_files[i])
        depth_image = o3d.io.read_image(depth_files[i])
        color_array.append(color_image)
        depth_array.append(depth_image)
        rgbd = create_one_RGBD(color_image, depth_image)
        rgbd_array.append(rgbd)

    # Perform pointcloud registration
    pose_graph, pc_list = registration(rgbd_array, intrinsic, pc_list)
    optimize_pose(pose_graph, 0.01, 0.1, 0.25, 0)

    # Collected all optimzied data and put it under one pointclouds.
    for node in range(len(pc_list)-1):
        pc_list[node].transform(pose_graph.nodes[node].pose)
        base += pc_list[node]
        base = o3d.geometry.PointCloud.voxel_down_sample(base, 0.005)

    mesh = surface_reconstruction(base, 0.1)

    # Show the result
    o3d.visualization.draw_geometries([mesh])


'''
RGBD INTEGRATION PIPELINE

    # create pose graph based on RGBD images
    pose_graph = create_RGBD_posegraph(5, color_files=color_array, depth_files=depth_array, intrinsics=intrinsic)
    pose_graph_optimization = optimize_pose(pose_graph, 0.01, 0.1, 0.25, 0)
    
    # Obtain mesh from RGBD surface reconstruction using TSDF
    mesh = integrate_RGBD(pose_graph_optimization, color_files, depth_files, 4.0)
    
    # Draw final result
    o3d.visualization.draw_geometries([mesh])
'''


