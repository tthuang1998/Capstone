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

        self.voxel_radius = 0.005
        self.max_corres_dist = 1.5
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(1280, 720, 601.1693115234375, 600.85931396484375,
                                                      637.83624267578125, 363.8018798828125)

    def start(self):
        self.sensor = o3d.io.AzureKinectSensor(self.config)
        if not self.sensor.connect(self.device):
            raise RuntimeError('Failed to connect to sensor')

    def escape_callback(self, vis):
        self.flag_exit = True
        return False

    def frames(self):
        while 1:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
            if rgbd is None:
                continue
            color, depth = np.asarray(rgbd.color).astype(np.uint8), np.asarray(rgbd.depth).astype(np.float32) / 1000.0
            return color, depth

    def run(self):

        glfw_key_escape = 256
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.create_window('viewer', 1920, 540)
        print("Sensor initialized. Press [ESC] to exit.")

        source_pcd_added = False

        while not self.flag_exit:

            rgbd = self.sensor.capture_frame(self.align_depth_to_color)

            if rgbd is None:
                continue

            color, depth = np.asarray(rgbd.color).astype(np.uint8), np.asarray(rgbd.depth).astype(np.float32) / 1000.0
            #color, depth = self.frames()

            depth = o3d.geometry.Image(depth)
            color = o3d.geometry.Image(color)

            rgbd_pcd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1.0, depth_trunc=0.50,
                                                                      convert_rgb_to_intensity=False)

            self.pcd_temp = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_pcd, self.intrinsics)
            #self.pcd_temp.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            self.pcd_temp, self.pc_fph = self.pointcloud_sampling()

            if not source_pcd_added:

                self.base_pcd = self.pcd_temp
                self.source_pcd = self.pcd_temp
                self.source_fph = self.pc_fph
                vis.add_geometry(self.base_pcd)
                #vis.add_geometry(rgbd)
                source_pcd_added = True
            else:
                #information_icp = self.registration(self.pc_fph)
                pose_graph = self.create_pose_graph()
                self.optimize_Pose(pose_graph, self.max_corres_dist)

                self.pcd_temp.transform(pose_graph.nodes[self.base_id].pose)
                self.base_pcd += self.pcd_temp

                self.source_pcd = copy.deepcopy(self.pcd_temp)
                self.source_fph = self.pc_fph

            #self.base_pcd = o3d.geometry.PointCloud.voxel_down_sample(self.base_pcd, voxel_size=0.2)

            #vis.add_geometry(self.base_pcd)
            vis.update_geometry(self.base_pcd)
            #vis.update_geometry(rgbd)
            vis.poll_events()
            vis.update_renderer()
            #vis.remove_geometry(self.base_pcd)

    def pointcloud_sampling(self):
        self.pcd_temp.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pc_temp = copy.deepcopy(self.pcd_temp)
        pc_temp = o3d.geometry.PointCloud.voxel_down_sample(pc_temp, voxel_size=self.voxel_radius)
        o3d.geometry.PointCloud.estimate_normals(pc_temp, o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        pc_fph = o3d.registration.compute_fpfh_feature(pc_temp, o3d.geometry.KDTreeSearchParamHybrid(radius=0.5 * 5.0,
                                                                                                     max_nn=30))
        return pc_temp, pc_fph

    def registration(self, pc_fph):
        source_temp = copy.deepcopy(self.source_pcd)
        target_temp = copy.deepcopy(self.pcd_temp)

        result = o3d.registration.registration_fast_based_on_feature_matching(source_temp, target_temp, self.source_fph,
                                                                              pc_fph,
                                                                              o3d.registration.FastGlobalRegistrationOption())

        icp_coarse = o3d.registration.registration_icp(
            source_temp, target_temp, self.voxel_radius * 15, result.transformation,
            o3d.registration.TransformationEstimationPointToPlane())

        icp_fine = o3d.registration.registration_icp(
            source_temp, target_temp, self.voxel_radius * 1.5, icp_coarse.transformation,
            o3d.registration.TransformationEstimationPointToPlane())

        information_icp = o3d.registration.get_information_matrix_from_point_clouds(
            source_temp, target_temp, self.voxel_radius * 1.5,
            icp_fine.transformation)

        transformation = icp_fine.transformation

        return transformation, information_icp

    def create_pose_graph(self):
        odometry = np.identity(4)
        pose_graph = o3d.registration.PoseGraph()

        trans, information_icp = self.registration(self.pc_fph)
        odometry = np.dot(trans, odometry)

        pose_graph.nodes.append(o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))

        pose_graph.edges.append(
            o3d.registration.PoseGraphEdge(self.base_id, self.target_id, self.current_trans, information_icp, uncertain=True))

        return pose_graph

    def optimize_Pose(self, pose_graph, max_corres_dist):
        option = o3d.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_corres_dist,
            edge_prune_threshold=0.25,
            reference_node=0)
        o3d.registration.global_optimization(
            pose_graph, o3d.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.registration.GlobalOptimizationConvergenceCriteria(), option)


cam = AzureKinect()
cam.start()
cam.run()
vis = o3d.visualization.draw_geometries([cam.base_pcd])
