"""
compute_gt_poses.py
---------------

Main Function for registering (aligning) colored point clouds with ICP/aruco marker
matching as well as pose graph optimizating, output transforms.npy in each directory

"""
import glob
import os

import numpy as np
from open3d import pipelines
from tqdm import trange

from Camera_Track import Camera_Track
from params import max_correspondence_distance_fine, LABEL_INTERVAL, N_Neighbours
from updatable_pose_graph import PoseGraph
from utils import load_images, load_pcd, make_target_frame_list
from utils import marker_registration


def full_registration(path, max_correspondence_distance_fine, camera_intrinsics, n_pcds):
    pose_graph = PoseGraph()
    camera_track = Camera_Track(pose_graph)

    pcds = [[] for i in range(n_pcds)]
    for source_id in trange(n_pcds):  # 对每一帧进行处理
        if source_id > 0:
            pcds[source_id - 1] = []
        target_id_list = make_target_frame_list(source_id, n_pcds)
        for target_id in target_id_list:  # source_id是当前帧，target_id是目标帧

            # derive pairwise registration through feature matching
            color_src, depth_src_cloudify, depth_src = load_images(path, source_id, camera_intrinsics)
            color_dst, depth_dst_cloudify, depth_dst = load_images(path, target_id, camera_intrinsics)

            if not pcds[source_id]:
                pcds[source_id] = load_pcd(path, source_id, camera_intrinsics, downsample=True)
            if not pcds[target_id]:
                pcds[target_id] = load_pcd(path, target_id, camera_intrinsics, downsample=True)

            if target_id == source_id + 1:  # odometry
                if source_id == 0:  # Init
                    camera_track.initialize(source_id, color_src, depth_src_cloudify, depth_src, pcds[source_id])
                camera_track.track(target_id, color_dst, depth_dst_cloudify, depth_dst, pcds[target_id])

            else:  # loop closure
                if abs(target_id - source_id) <= N_Neighbours:
                    use_keypoint = True
                else:
                    use_keypoint = False
                res = marker_registration((color_src, depth_src_cloudify),
                                          (color_dst, depth_dst_cloudify), use_SIFT_keypoint=use_keypoint)
                if res is None:
                    # ignore such connections
                    continue
                transformation_icp = res
                information_icp = pipelines.registration.get_information_matrix_from_point_clouds(
                    pcds[source_id], pcds[target_id], max_correspondence_distance_fine,
                    transformation_icp)
                pose_graph.Set_Edge(source_id, target_id, pipelines.registration.PoseGraphEdge(source_id, target_id,
                                                                                               transformation_icp,
                                                                                               information_icp,
                                                                                               uncertain=True))

    return pose_graph


def compute_camera_pose(folder_path, camera_intrinsics):
    Ts = []
    T_npy_path = os.path.join(folder_path, 'transform.npy')
    if os.path.exists(T_npy_path):
        print('transform file exist')
        return np.load(T_npy_path)
    else:
        n_pcds = int(len(glob.glob1(folder_path + "/rgb", "*.png")) / LABEL_INTERVAL)
        print("Full registration ...")
        pose_graph = full_registration(folder_path, max_correspondence_distance_fine, camera_intrinsics, n_pcds)

        print("Optimizing PoseGraph ...")
        pose_graph.optimize()

        num_annotations = int(len(glob.glob1(folder_path + "/rgb", "*.png")) / LABEL_INTERVAL)

        for point_id in range(num_annotations):
            Ts.append(pose_graph.get_pose(point_id))
        Ts = np.array(Ts)
        np.save(T_npy_path, Ts)
        print("Transforms saved")
        return Ts
