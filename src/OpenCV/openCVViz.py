'''
Authors: Alex Buck, Thomas Huang, Arjun Sree Manoj
Date: 10/27/2019

openCVViz: Run our pipeline with openCV library and perform registration
'''

import pyrealsense2 as rs2
import numpy as np
import cv2
import open3d as o3d
from opencv import initialize_opencv
import math
import copy
import realsense_device_manager as dev


# check opencv python package
with_opencv = initialize_opencv()

'''
Function: get intrinsic parameters of the device converted to open3D format

Param: 
frame -> frame from video streaming device

Output: open3D intrinsics in a matrix format used to position data in frame correctly
'''
def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)

    return out

'''
Function: Convert all depth and color images into pointclouds

Param: 
color_files -> rgb array from all frames collected
depth_files -> depth array from all frames collected
intrinsics -> intrinsic parameters of the device

Output: Array of pointclouds for all frames collected, len of the rgb array
'''
def process_rgbd_image(color_files, depth_files, intrinsic):
    pc_array = []
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    for frame in np.arange(0, len(color_files)):
        rgbd_image = o3d.create_rgbd_image_from_color_and_depth(color_files[frame], depth_files[frame],
                                                                depth_scale=1.0 / depth_scale,
                                                                depth_trunc=clipping_distance_in_meters,
                                                                convert_rgb_to_intensity=False)
        pc = o3d.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)
        pc.transform(flip_transform)
        o3d.estimate_normals(pc, o3d.geometry.KDTreeSearchParamHybrid(0.02, max_nn=30))
        pc_down = o3d.voxel_down_sample(pc, voxel_size=0.005)
        pc_array.append(pc_down)
    return pc_array, len(color_files)

'''
Function: Make fragments by combining sets of pointclouds together

Param: 
pc_array -> pointcloud array retrived from all frames collected
fragment_size -> number of pointclouds we are going to register to make a fragment

Output: Array of pointcloud fragments
'''
def make_fragment(pc_array, fragment_size):
    fragment = o3d.PointCloud()
    fragment_array = []
    count = 0
    for pc in pc_array:
        fragment += copy.deepcopy(pc)
        count += 1
        print(fragment)
        if count % fragment_size == 0:
            fragment_array.append(fragment)
            print("Fragment added")
            fragment.clear()

    return fragment_array, len(fragment_array)

'''
Function: Use all pointclouds to develop a pose graph for the object

Param: 
pc_array -> pointcloud array retrived from all frames collected
number_of_pc -> length of pointcloud array

Output: pose_graph of all pointclouds collected
'''
def process_pointclouds(pc_array, number_of_pc):

    voxel_size = 0.004
    icp_radius = voxel_size * 2

    current_trans = np.identity(4)
    odometry = np.identity(4)
    last_trans = np.identity(4)

    pose_graph = o3d.PoseGraph()
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))

    #Perform registration
    for source_id in np.arange(number_of_pc):
        for target_id in np.arange(source_id + 1, number_of_pc):

            source_down = copy.deepcopy(pc_array[source_id])
            target_down = copy.deepcopy(pc_array[target_id])

            if target_id == source_id + 1:
                print("reg node " + str(source_id) + " to " + str(target_id))

                result_icp = o3d.registration.registration_icp(source_down, target_down, icp_radius, current_trans,
                                                               o3d.registration.TransformationEstimationPointToPlane(),
                                                               o3d.registration.ICPConvergenceCriteria(
                                                                   max_iteration=100))

                information_icp = o3d.registration.get_information_matrix_from_point_clouds(source_down, target_down,
                                                                                            voxel_size * 1.5,
                                                                                            result_icp.transformation)
                current_trans = result_icp.transformation
                last_trans = result_icp.transformation

                odometry = np.dot(result_icp.transformation, odometry)

                pose_graph.nodes.append(o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))

                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(source_id, target_id, result_icp.transformation, information_icp,
                                                   uncertain=False))

            #register every nth pointcloud to the previous pointcloud fragment
            elif (source_id % 10 == 0) and (target_id % 10 == 0):
                print("reg edge of " + str(source_id) + " to " + str(target_id))

                result_icp = o3d.registration_icp(source_down, target_down, 0.02, last_trans, o3d.registration.TransformationEstimationPointToPlane(),
                                                           o3d.registration.ICPConvergenceCriteria(
                                                               max_iteration=100))

                information_icp = o3d.registration.get_information_matrix_from_point_clouds(source_down, target_down,
                                                                                            voxel_size * 1.5,
                                                                                            result_icp.transformation)

                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(source_id, target_id, result_icp.transformation, information_icp,
                                                   uncertain=True))
                last_trans = result_icp.transformation

    #Optimize pose graph - retrived from Open3D library
    option = o3d.registration.GlobalOptimizationOption(max_correspondence_distance=1.5,
                                                       edge_prune_threshold=0.25, reference_node=0,
                                                       preference_loop_closure=1)
    o3d.registration.global_optimization(pose_graph,
                                         o3d.registration.GlobalOptimizationLevenbergMarquardt(),
                                         o3d.registration.GlobalOptimizationConvergenceCriteria(),
                                         option)
    return pose_graph

'''
Function: Show the final stitched pointcloud

Param: 
pose_graph -> pose graph created from the pointclouds retrieved
fragment_array -> contains array of the pointcloud fragments registered together
number_of_fragment -> length of the fragment array

Output: draws pointcloud onto Open3D viewer
'''
def show_cloud(pose_graph, fragment_array, number_of_fragment):

    vis = o3d.VisualizerWithKeyCallback()
    vis.create_window('AFO', width=1280, height=720)
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])

    pcd_combined = o3d.PointCloud()

    for i in np.arange(number_of_fragment):
        target_temp = copy.deepcopy(fragment_array[i])
        cl, ind = o3d.geometry.statistical_outlier_removal(target_temp, nb_neighbors=50, std_ratio=.3)
        target_temp = o3d.geometry.select_down_sample(target_temp, ind)
        pcd_combined += target_temp.transform(pose_graph.nodes[i].pose)

    pcd_combined = o3d.geometry.voxel_down_sample(pcd_combined, voxel_size=0.001)
    cl, ind = o3d.geometry.statistical_outlier_removal(pcd_combined, nb_neighbors=50, std_ratio=.5)
    pcd_combined = o3d.geometry.select_down_sample(pcd_combined, ind)
    o3d.visualization.draw_geometries([pcd_combined])


#Initate IntelRealsense
pipeline = rs2.pipeline()

config = rs2.config()
config.enable_stream(rs2.stream.depth, 848, 480, rs2.format.z16, 30)
config.enable_stream(rs2.stream.color, 640, 480, rs2.format.bgr8, 30)

profile = pipeline.start(config)

sensor = profile.get_device()
depth_sensor = profile.get_device().first_depth_sensor()

depth_sensor.set_option(rs2.option.depth_units, 0.0001)
depth_sensor.set_option(rs2.option.visual_preset, 4)

depth_scale = depth_sensor.get_depth_scale()

# Truncate how far sensor can see
clipping_distance_in_meters = 0.7
clipping_distance = clipping_distance_in_meters/depth_scale

# Aligning depth frame with color frame
align_to = rs2.stream.color
align = rs2.align(align_to)

depth_image_array = []
color_image_array = []

try:
    while True:

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # filtered_frames = dec_filter.process(aligned_frames).as_frameset()

            # Get aligned frames
            filtered_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            intrinsic = get_intrinsic_matrix(color_frame)

            # Validate that both frames are valid
            if not filtered_depth_frame or not color_frame:
                continue

            depth_image = o3d.Image(np.asanyarray(filtered_depth_frame.get_data()))
            color_image = o3d.Image(np.asanyarray(color_frame.get_data()))

            depth_image_array.append(depth_image)
            color_image_array.append(color_image)

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            # depth image is 1 channel, color is 3 channels
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_removed = np.where(((depth_image_3d > clipping_distance) | (depth_image_3d <= 0)), grey_color, color_image)

            # Render images
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_3d, alpha=0.09), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))
            cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Recorder Realsense', images)
            key = cv2.waitKey(1)

            # if 'esc' button pressed, escape loop and exit program
            if key == 27:
                    cv2.destroyAllWindows()
                    pc_array, number_of_pc = process_rgbd_image(color_image_array, depth_image_array, intrinsic)
                    fragment_array, number_of_fragment = make_fragment(pc_array, 50)
                    pose_graph = process_pointclouds(fragment_array, number_of_fragment)
                    show_cloud(pose_graph, fragment_array, number_of_fragment)
                    break
finally:
    pipeline.stop()