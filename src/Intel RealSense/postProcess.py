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


# Get Cloud from rgbd image


def get_cloud(image, intrinsics):
    pc = o3d.create_point_cloud_from_rgbd_image(image, intrinsics)
    pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pc


# Setup cloud for registration
def setup_cloud(pointcloud):
    #qo3d.estimate_normals(pointcloud, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pc_fph = o3d.compute_fpfh_feature(pointcloud, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1 * 5.0, max_nn=30))
    return pointcloud, pc_fph

class VectorArrayInterface(object):
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def __array__(self, dtype=None):
        if dtype:
            return np.array([self.x, self.y, self.z], dtype=dtype)
        else:
            return np.array([self.x, self.y, self.z])


def capture_ply(pc, count):
    print("Capturing Point Cloud")
    o3d.write_point_cloud("C:/Users/rjsre/Desktop/Data Generated/data{}.ply".format(count), pc, write_ascii=True,
                          compressed=False)
    return


def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)

    return out

def odometry(source, target, intrinsic):

    #Initialize odometry parameters
    option = o3d.OdometryOption()
    odo_init = np.identity(4)

    #Compute odometry
    [success, trans, info] = o3d.compute_rgbd_odometry(source, target, intrinsic, odo_init, o3d.RGBDOdometryJacobianFromColorTerm(), option)

    return trans


# Remove bad points
def preprocess_point_cloud(pointcloud):
    pc_down = o3d.voxel_down_sample(pointcloud, voxel_size=0.01)
    o3d.estimate_normals(pc_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=50))
    o3d.orient_normals_towards_camera_location(pc_down,)
    return pc_down


def pairwise_registration(source, target):
    target_temp, target_f = setup_cloud(target)
    source_temp, source_f = setup_cloud(source)

    result = o3d.registration_fast_based_on_feature_matching(copy.deepcopy(source_temp), copy.deepcopy(target_temp), source_f, target_f,
                                                             o3d.FastGlobalRegistrationOption(maximum_correspondence_distance=0.01))
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.registration.registration_icp(
        copy.deepcopy(source_temp), copy.deepcopy(target_temp), voxel_radius * 1.5, result.transformation,
        o3d.TransformationEstimationPointToPlane(), o3d.ICPConvergenceCriteria(max_iteration=200))

    icp_fine = o3d.registration.registration_icp(
        copy.deepcopy(source_temp), copy.deepcopy(target_temp), voxel_radius * 1,
        icp_coarse.transformation,
        o3d.registration.TransformationEstimationPointToPlane(), o3d.ICPConvergenceCriteria(max_iteration=500))

    transformation_icp = icp_fine.transformation
    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        copy.deepcopy(source_temp), copy.deepcopy(target_temp), voxel_radius*1.5,
        icp_fine.transformation)

    if transformation_icp.trace() == 4.0:
        transformation_icp = np.identity(4)
        information_icp = np.zeros((6, 6))

    elif information_icp[5, 5] / min(len(copy.deepcopy(source_temp.points)), len(copy.deepcopy(target_temp.points))) < 0.3:
        transformation_icp = np.identity(4)
        information_icp = np.zeros((6, 6))

    return transformation_icp, information_icp


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

    return transformation_icp, information_icp


def add_pose_node(transformation, information, world_transformation, tid, pose_graph):

    for y in range(tid):
        if tid == y + 1:
            world_transformation = np.dot(transformation, world_transformation)
            node = o3d.PoseGraphNode(np.linalg.inv(world_transformation))
            oedge = o3d.PoseGraphEdge(y, tid, transformation, information, uncertain=False)
            pose_graph.nodes.append(node)
            pose_graph.edges.append(oedge)
        else:
            pose_graph.edges.append(o3d.PoseGraphEdge(y, tid, world_transformation, information, uncertain=True))
            #qpose_graph.edges.append(o3d.PoseGraphEdge(y, tid, transformation, information, uncertain=True))

    return pose_graph, world_transformation


def full_registration(source, target, source_id, target_id, base_id, base,  max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):

    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    odometry_global = np.identity(4)
    # pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))

    transformation_icp, information_icp = pairwise_registration(source, target)
    transformation_global, information_global = global_registration(base, source)

    transformation_global, success = error_checking(transformation_icp, transformation_global)

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

    trans_tol = 0.5  # transformation tolerance
    rotation_tol = 30  # 30 degrees
    # angle_with_pl_norm_tol = 0.087266 # in radians (= 5 degrees)
    # angle_with_pl_norm_tol = 0.174533 # in radians (= 10 degrees)
    angle_with_pl_norm_tol = 1.04  # in radians (= 20 degrees)
    if (tf[0, 3] > trans_tol or tf[0, 3] < -trans_tol or
            tf[1, 3] > trans_tol or tf[1, 3] < -trans_tol or
            tf[2, 3] > trans_tol or tf[2, 3] < -trans_tol):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("here in 1 ")
        return False
    #elif (theta_tot > rotation_tol or
     #     theta_tot < - rotation_tol):
      #  print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
       # print("here in 2 ")q
        #qreturn False
    else:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print("here in 4 ")q
        return True


def cal_angle(pl_norm, R_dir):
    angle_in_radians = \
        np.arccos(
            np.abs(pl_norm[0] * R_dir[0] + pl_norm[1] * R_dir[1] + pl_norm[2] * R_dir[2])
        )

    return angle_in_radians


def optimize_Pose(pose_graph, max_correspondence_distance_fine):
    option = o3d.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0, preference_loop_closure=2.0)
    o3d.registration.global_optimization(
        pose_graph, o3d.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.registration.GlobalOptimizationConvergenceCriteria(), option)






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
voxel_radius = 0.005
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
combined = o3d.PointCloud()

pose_graph = o3d.PoseGraph()
last_trans = np.identity(4)
check_trans = False

# Get IDS
sid = 0
tid = 1

pcd_list = []

#Attempt of RGBD Registration
rgbd_color = []
rgbd_depth = []

'''
# Initialize visualizer Class Config
vis = o3d.VisualizerWithKeyCallback()
vis.create_window('AFO', width=1280, height=720)
opt = vis.get_render_option()
opt.background_color = np.array([0, 0, 0])
# Streaming loop
'''

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
            pcd_list.append(target)

            current_trans, information_trans = pairwise_registration(copy.deepcopy(source), copy.deepcopy(target))

            check_trans = error_checking(worldTrans, current_trans)


            if check_trans:
                pose_graph, worldTrans = add_pose_node(current_trans, information_trans, worldTrans, tid, pose_graph)

                tid += 1
                print(tid)
                source = copy.deepcopy(target)

            else:
                pcd_list.remove(target)

        else:
            source = preprocess_point_cloud(get_cloud(rgbd_image, intrinsic))
            pcd_list.append(source)
            pose_graph.nodes.append(o3d.PoseGraphNode(np.identity(4)))
            save_image = True

        process_time = datetime.now() - dt0
        print("FPS: " + str(1 / process_time.total_seconds()))

        if keyboard.is_pressed('q'):
            optimize_Pose(pose_graph, max_correspondence_distance_fine=0.1)
            for node in range(len(pcd_list)):
                pcd_list[node].transform(pose_graph.nodes[node].pose)
                base += pcd_list[node]
            base = o3d.voxel_down_sample(base, voxel_size=0.005)
            # Initialize visualizer Class Configq
            vis = o3d.VisualizerWithKeyCallback()
            vis.create_window('AFO', width=1280, height=720)
            opt = vis.get_render_option()

            vis.add_geometry(base)
            vis.run()
            vis.destroy_window()
            capture_ply(base, 1)
            break

finally:
    pipeline.stop()