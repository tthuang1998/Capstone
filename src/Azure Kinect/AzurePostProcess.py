'''
Authors: Alex Buck, Thomas Huang, Arjun Sree Manoj
Date: 4/15/20

AzurePostProcess: Run our pipeline with post-scan processing using the Azure Kinect
'''

import argparse
import open3d as o3d
import numpy as np
import copy

class AzureKinect:
    def __init__(self):
        self.config = o3d.io.AzureKinectSensorConfig()
        self.device = 0
        self.align_depth_to_color = 1
        self.flag_exit = False

        self.source_pcd = o3d.geometry.PointCloud()
        self.base_pcd = o3d.geometry.PointCloud()
        self.pcd_temp = o3d.geometry.PointCloud()

        self.source_fph = o3d.registration.Feature()

        self.base_id = 0
        self.target_id = 1

        self.init_trans = np.identity(4)
        self.current_trans = np.identity(4)

        self.voxel_radius = 0.01
        self.max_corres_dist = 1.5
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(1280, 720, 601.1693115234375, 600.85931396484375,
                                                      637.83624267578125, 363.8018798828125)

        self.depth_array = []
        self.color_array = []
        self.rgbd_array = []

    def start(self):
        self.sensor = o3d.io.AzureKinectSensor(self.config)
        if not self.sensor.connect(self.device):
            raise RuntimeError('Failed to connect to sensor')

    def escape_callback(self, vis):
        self.flag_exit = True
        return False

    '''
    Function: split RGBD image into color & depth frames

    Param: self
    Output: color & depth frames
    '''    
    def frames(self):
        while 1:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
            if rgbd is None:
                continue
            color, depth = np.asarray(rgbd.color).astype(np.uint8), np.asarray(rgbd.depth).astype(np.float32) / 1000.0
            return color, depth

    '''
    Function: Run camera input until escape flag is called
    '''
    def run(self):

        glfw_key_escape = 256
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.create_window('viewer', 1920, 540)
        print("Sensor initialized. Press [ESC] to exit.")

        while not self.flag_exit:

            rgbd = self.sensor.capture_frame(self.align_depth_to_color)

            if rgbd is None:
                continue

            #self.rgbd_array.append(rgbd)


            color, depth = np.asarray(rgbd.color).astype(np.uint8), np.asarray(rgbd.depth).astype(np.float32) / 1000.0
            #color, depth = self.frames()

            depth = o3d.geometry.Image(depth)
            color = o3d.geometry.Image(color)

            self.depth_array.append(depth)
            self.color_array.append(color)


            vis.add_geometry(rgbd)
            #vis.add_geometry(self.base_pcd)
            #vis.update_geometry(self.base_pcd)
            vis.update_geometry(rgbd)
            vis.poll_events()
            vis.update_renderer()
            #vis.remove_geometry(self.base_pcd)

    '''
    Function: Downsamples & takes important features of pointcloud to be used for registration

    Output: downsampled pointcloud, pointcloud with distinct features
    '''
    def pointcloud_sampling(self):
        self.pcd_temp.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pc_temp = copy.deepcopy(self.pcd_temp)
        pc_temp = o3d.geometry.PointCloud.voxel_down_sample(pc_temp, voxel_size=self.voxel_radius)
        o3d.geometry.PointCloud.estimate_normals(pc_temp, o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        pc_fph = o3d.registration.compute_fpfh_feature(pc_temp, o3d.geometry.KDTreeSearchParamHybrid(radius=0.5 * 5.0,
                                                                                                     max_nn=30))
        return pc_temp, pc_fph

    '''
    Function: registers pointclouds to each other using fast, coarse, and fine registration

    Param: pc_fph -> pointcloud with features used for registration
    Output: tranformation pointcloud, information matrix from registration
    '''
    def registration(self, pc_fph):
        source_temp = copy.deepcopy(self.source_pcd)
        target_temp = copy.deepcopy(self.pcd_temp)

        result = o3d.registration.registration_fast_based_on_feature_matching(source_temp, target_temp, self.source_fph,
                                                                              pc_fph,
                                                                              o3d.registration.FastGlobalRegistrationOption())

        icp_coarse = o3d.registration.registration_icp(
            source_temp, target_temp, self.voxel_radius * 10, result.transformation,
            o3d.registration.TransformationEstimationPointToPlane())

        icp_fine = o3d.registration.registration_icp(
            source_temp, target_temp, self.voxel_radius * 1.0, icp_coarse.transformation,
            o3d.registration.TransformationEstimationPointToPlane())

        information_icp = o3d.registration.get_information_matrix_from_point_clouds(
            source_temp, target_temp, self.voxel_radius * 1.0,
            icp_fine.transformation)

        transformation = icp_fine.transformation

        return transformation, information_icp

    '''
    Function: creates pose graph from registered point cloud

    Output: pose graph created using transformation pointcloud and information matrix
    '''
    def create_pose_graph(self):
        odometry = np.identity(4)
        pose_graph = o3d.registration.PoseGraph()

        trans, information_icp = self.registration(self.pc_fph)
        odometry = np.dot(trans, odometry)

        pose_graph.nodes.append(o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))

        pose_graph.edges.append(
            o3d.registration.PoseGraphEdge(self.base_id, self.target_id, self.current_trans, information_icp, uncertain=True))

        pose_graph.edges.append(
            o3d.registration.PoseGraphEdge(self.base_id, self.target_id, self.current_trans, information_icp,
                                           uncertain=False))
        return pose_graph

    '''
    Function: optimizes pose graph

    Param: pose_graph -> pose graph constructed from the pointcloud
    max_corres_dist -> maximum correspondance distance between two nodes
    '''
    def optimize_Pose(self, pose_graph, max_corres_dist):
        option = o3d.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_corres_dist,
            edge_prune_threshold=0.50,
            reference_node=0)
        o3d.registration.global_optimization(
            pose_graph, o3d.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.registration.GlobalOptimizationConvergenceCriteria(), option)

    '''
    Function: processes images collected during run() & visualizes processing
    '''
    def postProcess(self):

        vis = o3d.visualization.Visualizer()
        vis.create_window('viewer', 1920, 1080)

        source_pcd_added = False

        for img in range(len(self.color_array)):

            #color, depth = np.asarray(self.color_array[img]).astype(np.uint8), np.asarray(self.depth_array[img]).astype(np.float32) / 1000.0

            rgbd_pcd = o3d.geometry.RGBDImage.create_from_color_and_depth(self.color_array[img], self.depth_array[img], depth_scale=1.0, depth_trunc=0.50,
                                                                     convert_rgb_to_intensity=False)

            self.pcd_temp = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_pcd, self.intrinsics)
            #self.pcd_temp.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            self.pcd_temp, self.pc_fph = self.pointcloud_sampling()

            if not source_pcd_added:

                self.base_pcd = self.pcd_temp
                self.source_pcd = self.pcd_temp
                self.source_fph = self.pc_fph
                vis.add_geometry(self.base_pcd)
                # vis.add_geometry(rgbd)
                source_pcd_added = True

            else:
                #information_icp = self.registration(self.pc_fph)
                pose_graph = self.create_pose_graph()
                self.optimize_Pose(pose_graph, self.max_corres_dist)

                self.pcd_temp.transform(pose_graph.nodes[self.base_id].pose)
                self.base_pcd += self.pcd_temp

                vis.add_geometry(self.base_pcd)
                vis.update_geometry(self.base_pcd)
                vis.poll_events()
                vis.update_renderer()

                self.source_pcd = copy.deepcopy(self.pcd_temp)
                self.source_fph = self.pc_fph



        #self.base_pcd = o3d.geometry.PointCloud.voxel_down_sample(self.base_pcd, voxel_size=0.2)

cam = AzureKinect()
cam.start()
cam.run()
cam.postProcess()
o3d.visualization.draw_geometries([cam.base_pcd])

# attempts surface reconstruction based on ball-pivoting algorithm
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cam.base_pcd, o3d.utility.DoubleVector([0.007, 0.007 *2]))
o3d.visualization.draw_geometries([mesh])
# attempts surface reconstruction based on Poisson algorithm
[mesh_p, vector_p] = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cam.base_pcd)
o3d.visualization.draw_geometries([mesh_p])