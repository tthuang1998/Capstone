'''
Authors: Alex Buck, Thomas Huang, Arjun Sree Manoj
Date: 11/13/20

realTime: Run our pipeline with real-time processing using the Intel RealSense
'''

from datetime import datetime
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import copy
import keyboard
from pynput.keyboard import Key
import scipy as sci
import modern_robotics as mr
# import pcl


'''
Function: Create pointcloud from RGBD image and camera intrinsics

Params: image -> RGBD image
intrinsics -> camera intrinsics

Output: pointcloud constructed from RGBD image
'''
def get_cloud(image, intrinsics):
    pc = o3d.create_point_cloud_from_rgbd_image(image, intrinsics)
    pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pc

'''
Function: Set up pointcloud for registration by estimating normals and computing key features

Params: pointcloud -> pointcloud constructed from RGBD image
Output: pointcloud with normals, pointcloud with features
'''
def setup_cloud(pointcloud):
    pc_temp = copy.deepcopy(pointcloud)
    o3d.estimate_normals(pc_temp, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pc_fph = o3d.compute_fpfh_feature(pc_temp, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1 * 5.0, max_nn=30))
    return pc_temp, pc_fph

'''
Function: get normals from a pointcloud

Params: pointcloud -> pointcloud with normals
Output: rotation direction for pointcloud computed from normals
'''
def get_normals(pointcloud):
    mean_normal_x = 0
    mean_normal_y = 0
    mean_normal_z = 0

    normal_count = 0

    for normal in pointcloud.normals:
        normal_count += 1
        mean_normal_x += normal[0]
        mean_normal_y += normal[1]
        mean_normal_z += normal[2]

    mean_normal_x /= normal_count
    mean_normal_y /= normal_count
    mean_normal_z /= normal_count

    rotation_dir = VectorArrayInterface(mean_normal_x, mean_normal_y, mean_normal_z).__array__()

    return rotation_dir

'''
Function: register pointclouds together

Params: source -> source pointcloud
target -> target pointcloud

Output: transformation pointcloud, information matrix from registration
'''
def registration(source, target):

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
    '''
    result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
        source_temp, target_temp, feature_source, feature_target, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    '''

    #Refine registration

    refine_result = o3d.registration.registration_icp(
        source_temp, target_temp, distance_threshold, result.transformation,
        o3d.registration.TransformationEstimationPointToPlane())

    information_icp = o3d.get_information_matrix_from_point_clouds(source_temp, target_temp, 0.01, refine_result.transformation)
    return refine_result.transformation, information_icp

    '''
    # Rotation Correction Algorithm
    tf = refine_result.transformation
    R = tf[:3, :3]
    so3mat = mr.MatrixLog3(R)
    omg = mr.so3ToVec(so3mat)
    R_dir, theta = mr.AxisAng3(omg)  # rotation direction, rotation angle (in radians)
    theta_degree = theta / np.pi * 180  # in degree
    angle_with_pl_norm = cal_angle(rotation_dir, R_dir)

    trans_tol = 0.5
    rotation_tol = 30
    angle_with_pl_norm_tol = 1.3

    if (tf[0, 3] > trans_tol or tf[0, 3] < -trans_tol or tf[1, 3] > trans_tol or tf[1, 3] < -trans_tol or
            tf[2, 3] > trans_tol or tf[2, 3] < -trans_tol):
        good = False
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("here in translation ")
        return np.identity(4), good

    elif (theta_degree > rotation_tol or theta_degree < -rotation_tol):
        good = False
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("here in rotation ")
        return np.identity(4), good
    elif angle_with_pl_norm > angle_with_pl_norm_tol:
        good = False
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("here in 3 ")
        print(" angle with pl norm")
        print(angle_with_pl_norm)
        return np.identity(4), good
    else:
        good = True
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        information_icp = o3d.get_information_matrix_from_point_clouds(source, target, 0.01, refine_result.transformation)
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


def cal_normal_angle(norm1, norm2):

    angle = np.arccos(np.abs(norm1[0]*norm2[0] + norm1[1]*norm2[1] + norm1[2]*norm2[2]) /
                     np.sqrt(norm1[0]*norm1[0] + norm1[1]*norm1[1] + norm1[2]*norm1[2]) /
                     np.sqrt(norm2[0]*norm2[0] + norm2[1]*norm2[1] + norm2[2]*norm2[2]))
    return angle

'''
Function: write pointcloud to file in .ply format

Params: pc -> pointcloud being saved
count -> frame
'''
def capture_ply(pc, count):
    print("Capturing Point Cloud")
    o3d.write_point_cloud("C:/Users/rjsre/Desktop/Data Generated/data{}.ply".format(count), pc, write_ascii=True,
                          compressed=False)
    return

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
Function: Downsample pointcloud and remove outliers

Param: 
pointcloud -> self-explanatory

Output: processed pointcloud
'''
def preprocess_point_cloud(pointcloud):
    pc_down = o3d.voxel_down_sample(pointcloud, voxel_size=0.004)
    o3d.estimate_normals(pc_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    o3d.orient_normals_towards_camera_location(pc_down,)

    return pc_down

'''
Function: register two subsequent pointclouds

Param: 
source -> source Pointcloud, initial transformation established from here
target -> target Pointcloud, trying to transform to fit to source Pointcloud

Output: transformation matrix and information matrix on how to transform target to fit source
'''
def pairwise_registration(source, target):
    target_temp, target_f = setup_cloud(target)
    source_temp, source_f = setup_cloud(source)

    result = o3d.registration_fast_based_on_feature_matching(source_temp, target_temp, source_f, target_f,
                                                             o3d.FastGlobalRegistrationOption())
    print("Apply point-to-plane ICP")
    #icp_coarse = o3d.registration.registration_icp(
     #   source_temp, target_temp, voxel_radius * 15, result.transformation,
     #   o3d.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.registration.registration_icp(
        source_temp, target_temp, voxel_radius * 1.5,
        result.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    #transformation_icp = result.transformation
    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        source_temp, target_temp, voxel_radius*1.5,
        transformation_icp)
    return transformation_icp, information_icp

'''
Function: register two pointclouds that may not be in close proximity

Param: 
source -> source Pointcloud, initial transformation established from here
target -> target Pointcloud, trying to transform to fit to source Pointcloud

Output: transformation matrix and information matrix on how to transform target to fit source
'''
def global_registration(source, target):
    target_temp, target_f = setup_cloud(target)
    source_temp, source_f = setup_cloud(source)

    result = o3d.registration_fast_based_on_feature_matching(source_temp, target_temp, source_f, target_f,
                                                             o3d.FastGlobalRegistrationOption())

    #result = o3d.registration_icp(source_temp, target_temp, voxel_radius*1.5, np.identity(4),
                                  #o3d.TransformationEstimationPointToPlane())

    transformation_icp = result.transformation

    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        source_temp, target_temp, voxel_radius*1.5,
        transformation_icp)

    if transformation_icp.trace() == 4.0:
        transformation_icp = np.identity(4)
        information_icp = np.zeros((6, 6))

    elif information_icp[5, 5] / min(len(copy.deepcopy(source_temp.points)), len(copy.deepcopy(target_temp.points))) < 0.3:
        transformation_icp = np.identity(4)
        information_icp = np.zeros((6, 6))

    return transformation_icp, information_icp

'''
Function: conduct both pairwise and global registration for all pointclouds

Param: 
source -> source Pointcloud, initial transformation established from here
target -> target Pointcloud, trying to transform to fit to source Pointcloud
source_id -> tag for source pointcloud
target_id -> tag for target pointcloud
base_id -> tag for base pointcloud
base -> base pointcloud - holds all registered, accepted pointclouds

Output: final pose graph and boolean whether it could perform registration with the received pointclouds
'''
def full_registration(source, target, source_id, target_id, base_id, base,  max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):

    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    odometry_global = np.identity(4)
    success = True
    # pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))

    transformation_icp, information_icp = pairwise_registration(source, target)
    transformation_global, information_global = global_registration(base, source)

    #transformation_global, success = error_checking(transformation_icp, transformation_global)

    information_global = o3d.registration.get_information_matrix_from_point_clouds(
        source, target, voxel_radius * 1.5,
        transformation_global)

    odometry = np.dot(transformation_icp, odometry)
    odometry_global = np.dot(transformation_global, odometry_global)

    pose_graph.nodes.append(o3d.PoseGraphNode(np.linalg.inv(odometry)))

    if success:
        # Odometry Case for local registration
        pose_graph.edges.append(
            o3d.PoseGraphEdge(source_id, target_id, transformation_icp, information_icp, uncertain=False))
        pose_graph.edges.append(
            o3d.PoseGraphEdge(base_id, target_id, transformation_icp, information_icp, uncertain=True))

    # Loop Closure Case for global registration
    # pose_graph.nodes.append(o3d.PoseGraphNode(np.linalg.inv(odometry_global)))
    # pose_graph.edges.append(o3d.PoseGraphEdge(base_id, target_id, transformation_global, information_global, uncertain=True))

    return pose_graph, success
'''
(NOT WORKING)

Function: check for transformation error between registration processes

Param: 
transformation_prev -> previously stored transformation matrix from previous successful registration
transformation_new -> transformation matrix retrieved from current registration process

Output: return boolean of whether transformation is too far off from the previous one
'''
def error_checking(transformation_prev, transformation_new):
    tf = transformation_new
    R = tf[:3, :3]  # rotation matrix

    tf_prev = transformation_prev
    r = tf_prev[:3, :3]

    so3mat = mr.MatrixLog3(R)
    so3mat_prev = mr.MatrixLog3(r)

    omg_prev = mr.so3ToVec(so3mat_prev)
    omg = mr.so3ToVec(so3mat)

    R_dir, theta = mr.AxisAng3(omg)  # rotation direction
    rotation_dir, theta_prev = mr.AxisAng3(omg_prev)

    # rotation angle (in radians)
    theta_degree = theta / np.pi * 180  # in degree
    theta_degree_prev = theta_prev / np.pi * 180

    theta_tot = theta_degree - theta_degree_prev

    angle_with_pl_norm = cal_angle(rotation_dir, R_dir)

    trans_tol = 2.0 # transformation tolerance
    rotation_tol = 60  # 30 degrees
    # angle_with_pl_norm_tol = 0.087266 # in radians (= 5 degrees)
    # angle_with_pl_norm_tol = 0.174533 # in radians (= 10 degrees)
    angle_with_pl_norm_tol = 0.35  # in radians (= 20 degrees)
    if (tf[0, 3] > trans_tol or tf[0, 3] < -trans_tol or
            tf[1, 3] > trans_tol or tf[1, 3] < -trans_tol or
            tf[2, 3] > trans_tol or tf[2, 3] < -trans_tol):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("here in 1 ")
        return np.identity(4), False
    elif (theta_tot > rotation_tol or
          theta_tot < - rotation_tol):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("here in 2 ")
        return np.identity(4), False
    else:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print("here in 4 ")
        return transformation_new, True

'''
(NOT WORKING)

Function: calculate angle between rotation and normal computed

Param: 
pl_norm -> normal of the plane in which the pointcloud resides
R_dir -> rotation matrix direction

Output: angle in radians
'''
def cal_angle(pl_norm, R_dir):
    angle_in_radians = \
        np.arccos(
            np.abs(pl_norm[0] * R_dir[0] + pl_norm[1] * R_dir[1] + pl_norm[2] * R_dir[2])
        )

    return angle_in_radians

'''
Function: optimze pose graph using Open3D optimization functions for pose graphs

Param: 
pose_graph -> pose graph created from all pointclouds
max_correspondence_distance_fine -> maximum distance allowed between each frame registered, tigthen registration

Output: None
'''
def optimize_Pose(pose_graph, max_correspondence_distance_fine):
    option = o3d.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.registration.global_optimization(
        pose_graph, o3d.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.registration.GlobalOptimizationConvergenceCriteria(), option)

'''
Function: add key control functionality to open3D viewer

Param: 
pcd -> pointcloud generated

Output: Return visualizer if true, else None
'''
def custom_draw_geometry_with_key_callback(pcd):

    def show(vis):
        vis.add_geometry(pcd)
        vis.run()
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = show
    o3d.draw_geometries_with_key_callbacks([base], key_to_callback)



# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# New Module Test Try
check = False
good = True
rotation = False
rotation_dir = VectorArrayInterface(0, 0, 0).__array__()
count = 0

worldTrans = np.identity(4)
localTrans = np.identity(4)

# config of registration
save_image = False
voxel_radius = 0.004
max_iter = 15

# Start streaming
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
laserpwr = depth_sensor.get_option(rs.option.laser_power)
depth_sensor.set_option(rs.option.emitter_enabled, 1)
depth_sensor.set_option(rs.option.laser_power, laserpwr)
depth_sensor.set_option(rs.option.gain, 16)
depth_sensor.set_option(rs.option.depth_units, 0.0001)
depth_sensor.set_option(rs.option.visual_preset, 4)
depth_scale = depth_sensor.get_depth_scale()

# Truncate how far sensor can see
clipping_distance_in_meters = 0.75
clipping_distance = clipping_distance_in_meters/depth_scale

# Aligning depth frame with color frame
align_to = rs.stream.color
align = rs.align(align_to)

# Initialize pointClouds used
source = o3d.PointCloud()
target = o3d.PointCloud()

feature_source = o3d.Feature()
feature_target = o3d.Feature()

base = o3d.PointCloud()

# Get IDS
ids = 0
idt = 1
idb = 0

#Attempt TSDF Volume
volume = o3d.ScalableTSDFVolume(voxel_length=4.0/512.0, sdf_trunc=0.04, color_type=o3d.TSDFVolumeColorType.RGB8)
mesh = o3d.TriangleMesh()

#Attempt of RGBD Registration
rgbd_color = []
rgbd_depth = []


# Initialize visualizer Class Config
vis = o3d.VisualizerWithKeyCallback()
vis.create_window('AFO', width=1280, height=720)
opt = vis.get_render_option()
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

        # Need to update intrinsics of camera since position changes

        # Make sure frames come in together
        if not aligned_depth_frame or not color_frame:  # or not depth_2 or not color_2:
            continue

        # Create Images
        depth_image = o3d.Image(np.array(aligned_depth_frame.get_data()))
        color_temp = np.asarray(color_frame.get_data())
        color_image = o3d.Image(color_temp)

        # Create RGBD Image
        rgbd_image = o3d.create_rgbd_image_from_color_and_depth(color_image, depth_image, depth_scale=1.0 / depth_scale,
                                                                depth_trunc=clipping_distance_in_meters,
                                                                convert_rgb_to_intensity=False)

        if save_image:
            target = preprocess_point_cloud(get_cloud(rgbd_image, intrinsic))

            pose_graph, value = full_registration(source, target, source_id=ids, target_id=idt, base_id=idb, base=base,
                              max_correspondence_distance_fine=1.5,
                              max_correspondence_distance_coarse=15)

            if value:
                optimize_Pose(pose_graph, max_correspondence_distance_fine=1.5)
                target.transform(pose_graph.nodes[0].pose)
                base += target
                source = copy.deepcopy(target)
            else:
                # source = copy.deepcopy(base)
                continue

            base = o3d.voxel_down_sample(base, voxel_radius)

        else:
            base = preprocess_point_cloud(get_cloud(rgbd_image, intrinsic))
            source = copy.deepcopy(base)
            save_image = True

        vis.add_geometry(base)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        vis.remove_geometry(base)


        #capture_ply(base, count)

        process_time = datetime.now() - dt0
        print("FPS: " + str(1 / process_time.total_seconds()))

        if keyboard.is_pressed('q'):
            # Initialize visualizer Class Configq
            vis2 = o3d.VisualizerWithKeyCallback()
            vis2.create_window('AFO', width=1280, height=720)
            opt = vis2.get_render_option()
            opt.background_color = np.array([0, 0, 0])

            vis2.add_geometry(base)
            vis2.run()
            vis2.destroy_window()
            break

finally:
    pipeline.stop()