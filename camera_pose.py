"""
compute_gt_poses.py
---------------

Main Function for registering (aligning) colored point clouds with ICP/aruco marker
matching as well as pose graph optimizating, output transforms.npy in each directory

"""
import glob
import os

import cv2
import cv2.aruco as aruco
import numpy as np
from open3d import pipelines
from tqdm import trange

from utils import multiscale_icp, match_ransac, load_images, load_pcd, feature_registration, make_target_frame_list
from params import max_correspondence_distance_coarse, max_correspondence_distance_fine, VOXEL_SIZE, ICP_METHOD, \
    LABEL_INTERVAL, N_Neighbours


def marker_registration(source, target, use_SIFT_keypoint):
    cad_src, depth_src = source
    cad_des, depth_des = target

    gray_src = cv2.cvtColor(cad_src, cv2.COLOR_RGB2GRAY)
    gray_des = cv2.cvtColor(cad_des, cv2.COLOR_RGB2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    # lists of ids and the corners beloning to each id
    corners_src, _ids_src, rejectedImgPoints = aruco.detectMarkers(gray_src, aruco_dict, parameters=parameters)
    corners_des, _ids_des, rejectedImgPoints = aruco.detectMarkers(gray_des, aruco_dict, parameters=parameters)

    if use_SIFT_keypoint:
        feature_src_good, feature_dst_good = feature_registration(source, target)
    else:
        feature_src_good = []
        feature_dst_good = []
    assert len(feature_src_good) == len(feature_dst_good)

    try:
        ids_src = []
        ids_des = []
        for i in range(len(_ids_src)):
            ids_src.append(_ids_src[i][0])
        for i in range(len(_ids_des)):
            ids_des.append(_ids_des[i][0])
    except:
        return None

    common = [x for x in ids_src if x in ids_des]

    if (len(common) + len(feature_src_good)) < 12:
        # too few marker matches, use icp instead
        return None

    src_good = []
    dst_good = []
    for i, id in enumerate(ids_des):
        if id in ids_src:
            j = ids_src.index(id)
            for count, corner in enumerate(corners_src[j][0]):
                feature_3D_src = depth_src[int(corner[1])][int(corner[0])]
                feature_3D_des = depth_des[int(corners_des[i][0][count][1])][int(corners_des[i][0][count][0])]
                if feature_3D_src[2] != 0 and feature_3D_des[2] != 0:
                    src_good.append(feature_3D_src)
                    dst_good.append(feature_3D_des)

    src_good += feature_src_good
    dst_good += feature_dst_good

    # get rigid transforms between 2 set of feature points through ransac
    try:
        transform = match_ransac(np.asarray(src_good), np.asarray(dst_good))
        return transform
    except:
        return None


def full_registration(path, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine, camera_intrinsics, n_pcds):
    pose_graph = pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(pipelines.registration.PoseGraphNode(odometry))

    pcds = [[] for i in range(n_pcds)]
    for source_id in trange(n_pcds):  # ????????????????????????
        if source_id > 0:
            pcds[source_id - 1] = []
        target_id_list = make_target_frame_list(source_id, n_pcds)
        for target_id in target_id_list:  # source_id???????????????target_id????????????

            # derive pairwise registration through feature matching
            color_src, depth_src = load_images(path, source_id, camera_intrinsics)
            color_dst, depth_dst = load_images(path, target_id, camera_intrinsics)
            if abs(target_id - source_id) <= 18:
                use_keypoint = True
            else:
                use_keypoint = False
            res = marker_registration((color_src, depth_src),
                                      (color_dst, depth_dst), use_keypoint)

            if res is None and target_id != source_id + 1:
                # ignore such connections
                continue

            if not pcds[source_id]:
                pcds[source_id] = load_pcd(path, source_id, camera_intrinsics, downsample=True)
            if not pcds[target_id]:
                pcds[target_id] = load_pcd(path, target_id, camera_intrinsics, downsample=True)
            if res is None:
                # if marker_registration fails, perform pointcloud matching
                transformation_icp, information_icp = multiscale_icp(
                    pcds[source_id], pcds[target_id], [VOXEL_SIZE * 10, VOXEL_SIZE], [90, 90],
                    ["point_to_plane", "point_to_plane"])

            else:
                transformation_icp = res
                information_icp = pipelines.registration.get_information_matrix_from_point_clouds(
                    pcds[source_id], pcds[target_id], max_correspondence_distance_fine,
                    transformation_icp)

            if target_id == source_id + 1:
                # odometry
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(pipelines.registration.PoseGraphEdge(source_id, target_id,
                                                                             transformation_icp, information_icp,
                                                                             uncertain=False))
            else:
                # loop closure
                pose_graph.edges.append(pipelines.registration.PoseGraphEdge(source_id, target_id,
                                                                             transformation_icp, information_icp,
                                                                             uncertain=True))

    return pose_graph  # ??????????????????????????????SLAM????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????


def compute_camera_pose(folder_path, camera_intrinsics):
    Ts = []
    T_npy_path = os.path.join(folder_path, 'transform.npy')
    if os.path.exists(T_npy_path):
        print('transform file exist')
        return np.load(T_npy_path)
    else:
        n_pcds = int(len(glob.glob1(folder_path + "/rgb", "*.png")) / LABEL_INTERVAL)
        print("Full registration ...")
        pose_graph = full_registration(folder_path, max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine, camera_intrinsics, n_pcds)

        print("Optimizing PoseGraph ...")
        option = pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0)
        pipelines.registration.global_optimization(pose_graph,
                                                   pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                                   pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                   option)

        num_annotations = int(len(glob.glob1(folder_path + "/rgb", "*.png")) / LABEL_INTERVAL)

        for point_id in range(num_annotations):
            Ts.append(pose_graph.nodes[point_id].pose)
        Ts = np.array(Ts)
        np.save(T_npy_path, Ts)
        print("Transforms saved")
        return Ts
