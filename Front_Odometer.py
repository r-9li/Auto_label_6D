import numpy as np
from open3d import pipelines

from utils import marker_registration
from params import VOXEL_SIZE, max_correspondence_distance_fine
from utils import multiscale_icp


class Front_Odometer:
    def __init__(self, PoseGraph):
        self.pose_graph = PoseGraph
        self.odom = np.identity(4)
        self.last_rgb = None  # Source RGB image
        self.last_depth = None  # Source Depth
        self.last_pcd = None  # Source PCD
        self.last_index = None  # Source index

    def initialize(self, rgb_image, depth, pcd):
        self.last_rgb = rgb_image
        self.last_depth = depth
        self.last_pcd = pcd
        self.last_index = 0

        self.pose_graph.Set_Node(self.last_index, pipelines.registration.PoseGraphNode(np.linalg.inv(self.odom)))

    def track(self, rgb_image, depth, pcd, index):
        res = marker_registration((self.last_rgb, self.last_depth),
                                  (rgb_image, depth), use_SIFT_keypoint=True)

        if res is None:
            # if marker_registration fails, perform pointcloud matching
            transformation_icp, information_icp = multiscale_icp(
                self.last_pcd, pcd, [VOXEL_SIZE * 10, VOXEL_SIZE], [90, 90],
                ["point_to_plane", "point_to_plane"])
        else:
            transformation_icp = res
            information_icp = pipelines.registration.get_information_matrix_from_point_clouds(
                self.last_pcd, pcd, max_correspondence_distance_fine,
                transformation_icp)

        self.odom = np.dot(transformation_icp, self.odom)
        self.pose_graph.Set_Node(index, pipelines.registration.PoseGraphNode(np.linalg.inv(self.odom)))
        self.pose_graph.Set_Edge(self.last_index, index, pipelines.registration.PoseGraphEdge(self.last_index, index,
                                                                                              transformation_icp,
                                                                                              information_icp,
                                                                                              uncertain=False))
        self.last_rgb = rgb_image
        self.last_depth = depth
        self.last_pcd = pcd
        self.last_index = index

        return np.linalg.inv(self.odom)
