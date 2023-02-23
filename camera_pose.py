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

from utils import icp, match_ransac, load_images, load_pcd

'''
===============================================================================
Define a set of parameters related to fragment registration
===============================================================================
'''
# Voxel size used to down sample the raw pointcloud for faster ICP
VOXEL_SIZE = 0.001

# Set up parameters for post-processing
# Voxel size for the complete mesh
VOXEL_R = 0.0002

# search for up to N frames for registration, odometry only N=1, all frames N = np.inf
# for any N!= np.inf, the refinement is local
K_NEIGHBORS = 10

# Specify an icp algorithm
# "colored-icp", as in Park, Q.-Y. Zhou, and V. Koltun, Colored Point Cloud Registration Revisited, ICCV, 2017 (slower)
# "point-to-plane", a coarse to fine implementation of point-to-plane icp (faster)

ICP_METHOD = "point-to-plane"

# specify the frenquency of labeling ground truth pose

LABEL_INTERVAL = 1

# specify the frenquency of segments used in mesh reconstruction

RECONSTRUCTION_INTERVAL = 10

# Set up parameters for registration
# voxel sizes use to down sample raw pointcloud for fast ICP
voxel_size = VOXEL_SIZE
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

# Set up parameters for post-processing
# Voxel size for the complete mesh
voxel_Radius = VOXEL_R

# Point considered an outlier if more than inlier_Radius away from other points
inlier_Radius = voxel_Radius * 2.5

# search for up to N frames for registration, odometry only N=1, all frames N = np.inf
N_Neighbours = K_NEIGHBORS


def marker_registration(source, target):
    cad_src, depth_src = source
    cad_des, depth_des = target

    gray_src = cv2.cvtColor(cad_src, cv2.COLOR_RGB2GRAY)
    gray_des = cv2.cvtColor(cad_des, cv2.COLOR_RGB2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    # lists of ids and the corners beloning to each id
    corners_src, _ids_src, rejectedImgPoints = aruco.detectMarkers(gray_src, aruco_dict, parameters=parameters)
    corners_des, _ids_des, rejectedImgPoints = aruco.detectMarkers(gray_des, aruco_dict, parameters=parameters)
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

    if len(common) < 2:
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

    # get rigid transforms between 2 set of feature points through ransac
    try:
        transform = match_ransac(np.asarray(src_good), np.asarray(dst_good))
        return transform
    except:
        return None


def full_registration(path, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine, camera_intrinsics, n_pcds):
    global N_Neighbours, LABEL_INTERVAL
    pose_graph = pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(pipelines.registration.PoseGraphNode(odometry))

    pcds = [[] for i in range(n_pcds)]
    for source_id in trange(n_pcds):  # 对每一帧进行处理
        if source_id > 0:
            pcds[source_id - 1] = []
        # for target_id in range(source_id + 1, min(source_id + N_Neighbours,n_pcds)):
        for target_id in range(source_id + 1, n_pcds,
                               max(1, int(n_pcds / N_Neighbours))):  # source_id是当前帧，target_id是目标帧

            # derive pairwise registration through feature matching
            color_src, depth_src = load_images(path, source_id, camera_intrinsics)
            color_dst, depth_dst = load_images(path, target_id, camera_intrinsics)
            res = marker_registration((color_src, depth_src),
                                      (color_dst, depth_dst))

            if res is None and target_id != source_id + 1:
                # ignore such connections
                continue

            if not pcds[source_id]:
                pcds[source_id] = load_pcd(path, source_id, camera_intrinsics, downsample=True)
            if not pcds[target_id]:
                pcds[target_id] = load_pcd(path, target_id, camera_intrinsics, downsample=True)
            if res is None:
                # if marker_registration fails, perform pointcloud matching
                transformation_icp, information_icp = icp(
                    pcds[source_id], pcds[target_id], voxel_size, max_correspondence_distance_coarse,
                    max_correspondence_distance_fine, method=ICP_METHOD)

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

    return pose_graph  # 返回的是一个位姿图（SLAM里的概念），位姿图的节点数就是拍摄的图片数，边就是各个节点之间的变换矩阵。这个方法不止与相邻帧建立变换矩阵，还与一些不相邻的帧建立了，可能这样会更精确。


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
