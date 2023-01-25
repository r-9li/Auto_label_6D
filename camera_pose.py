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
import g2o
import numpy as np
import open3d
import png
from tqdm import trange

import pose_graph_optimization
from utils import poseRt

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

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
IMAGE_NUM = 10
MARKER_LENGTH = 0.055
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

mtx = np.array(0)

DEBUG = False


def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics):
    """
    Convert the depthmap to a 3D point cloud
    Parameters:
    -----------
    depth_frame : (m,n) uint16
            The depth_frame containing the depth map

    camera_intrinsics : dict
            The intrinsic values of the depth imager in whose coordinate system the depth_frame is computed
    Return:
    ----------
    pointcloud : (m,n,3) float
            The corresponding pointcloud in meters

    """

    [height, width] = depth_image.shape

    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() -
         float(camera_intrinsics['ppx'])) / float(camera_intrinsics['fx'])
    y = (v.flatten() -
         float(camera_intrinsics['ppy'])) / float(camera_intrinsics['fy'])
    depth_image = depth_image * float(camera_intrinsics['depth_scale'])
    z = depth_image.flatten()
    x = np.multiply(x, z)
    y = np.multiply(y, z)

    pointcloud = np.dstack((x, y, z)).reshape(
        (depth_image.shape[0], depth_image.shape[1], 3))

    return pointcloud


def marker_registration(source, target):
    global mtx
    cad_src = source
    cad_des = target

    gray_src = cv2.cvtColor(cad_src, cv2.COLOR_RGB2GRAY)
    gray_des = cv2.cvtColor(cad_des, cv2.COLOR_RGB2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    # Refine for higher precision
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    parameters.cornerRefinementMaxIterations = 90
    parameters.cornerRefinementMinAccuracy = 0.01

    # lists of ids and the corners beloning to each id
    _corners_src, _ids_src, rejectedImgPoints = aruco.detectMarkers(gray_src, aruco_dict, parameters=parameters)
    _corners_des, _ids_des, rejectedImgPoints = aruco.detectMarkers(gray_des, aruco_dict, parameters=parameters)
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

    if len(common) < 1:
        # too few marker matches
        print('too few marker matches, Failed!')
        return None
    corners_src = []
    corners_des = []
    for id in common:
        for i in range(len(_ids_src)):
            if _ids_src[i][0] == id:
                corners_src.append(_corners_src[i])
        for i in range(len(_ids_des)):
            if _ids_des[i][0] == id:
                corners_des.append(_corners_des[i])
    corners_src = tuple(corners_src)
    corners_des = tuple(corners_des)

    _rvec_src = []
    _tvec_src = []
    _rvec_des = []
    _tvec_des = []
    pnp_Method = cv2.SOLVEPNP_IPPE
    for i in range(len(common)):
        objp = np.array([[-MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0.0], [MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0.0],
                         [MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0.0], [-MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0.0]])
        _, _rvec_src_temp, _tvec_src_temp = cv2.solvePnP(objp, corners_src[i][0], mtx,
                                                         np.array(([[0., 0., 0., 0., 0.]])), flags=pnp_Method)
        _, _rvec_des_temp, _tvec_des_temp = cv2.solvePnP(objp, corners_des[i][0], mtx,
                                                         np.array(([[0., 0., 0., 0., 0.]])), flags=pnp_Method)
        _rvec_src_temp, _tvec_src_temp = cv2.solvePnPRefineVVS(objp, corners_src[i][0], mtx,
                                                               np.array(([[0., 0., 0., 0., 0.]])), _rvec_src_temp,
                                                               _tvec_src_temp)
        _rvec_des_temp, _tvec_des_temp = cv2.solvePnPRefineVVS(objp, corners_des[i][0], mtx,
                                                               np.array(([[0., 0., 0., 0., 0.]])), _rvec_des_temp,
                                                               _tvec_des_temp)
        _rvec_src.append(_rvec_src_temp)
        _tvec_src.append(_tvec_src_temp)
        _rvec_des.append(_rvec_des_temp)
        _tvec_des.append(_tvec_des_temp)
    _rvec_src = np.array(_rvec_src)
    _tvec_src = np.array(_tvec_src)
    _rvec_des = np.array(_rvec_des)
    _tvec_des = np.array(_tvec_des)

    if DEBUG:
        for i in range(_rvec_src.shape[0]):
            cv2.drawFrameAxes(cad_src, mtx, np.array(([[0., 0., 0., 0., 0.]])), _rvec_src[i, :, :], _tvec_src[i, :, :],
                              0.03)
            cv2.drawFrameAxes(cad_des, mtx, np.array(([[0., 0., 0., 0., 0.]])), _rvec_des[i, :, :], _tvec_des[i, :, :],
                              0.03)
            aruco.drawDetectedMarkers(cad_src, corners_src)
            aruco.drawDetectedMarkers(cad_des, corners_des)
        cv2.imshow('src', cad_src)
        cv2.imshow('des', cad_des)
        key = cv2.waitKey(0)
    trans_matrix_src_to_des = []
    for i in range(_rvec_src.shape[0]):
        trans_matrix_src = poseRt(cv2.Rodrigues(_rvec_src[i])[0], _tvec_src[i].squeeze())
        trans_matrix_des = poseRt(cv2.Rodrigues(_rvec_des[i])[0], _tvec_des[i].squeeze())
        trans_matrix_src_to_des.append(np.dot(np.linalg.inv(trans_matrix_src), trans_matrix_des))
        # (S_TO_D)=(A_TO_S).inv*(A_TO_D)
    optimize_list = []
    for i in range(len(trans_matrix_src_to_des)):
        optimize_r = cv2.Rodrigues(trans_matrix_src_to_des[i][:3, :3])[0].squeeze().tolist()
        optimize_t = trans_matrix_src_to_des[i][:3, 3].tolist()
        optimize_list.append(optimize_r + optimize_t)
    # Remove outliers
    if len(trans_matrix_src_to_des) == 1:
        T = trans_matrix_src_to_des[0]
    else:  # common Aruco >1
        model_iso = IsolationForest()
        preds = model_iso.fit_predict(optimize_list)
        for i in range(preds.shape[0] - 1, -1, -1):
            if preds[i] == -1:
                optimize_list.remove(optimize_list[i])
        model_kmeans = KMeans(n_clusters=1)
        model_kmeans.fit(optimize_list)
        optimized_pose = model_kmeans.cluster_centers_.squeeze()
        T = poseRt(cv2.Rodrigues(optimized_pose[:3])[0], optimized_pose[3:])
    return T


def full_registration(path, camera_intrinsics, n_pcds):
    odometry = np.identity(4)
    pose_graph = pose_graph_optimization.PoseGraphOptimization()
    pose_graph.add_vertex(0, g2o.Isometry3d(odometry), fixed=True)

    pcds = [[] for i in range(n_pcds)]
    for source_id in trange(n_pcds):  # 对每一帧进行处理
        if source_id > 0:
            pcds[source_id - 1] = []
        for target_id in range(source_id + 1, min(n_pcds, source_id + IMAGE_NUM)):

            # derive pairwise registration through feature matching
            color_src, depth_src = load_images(path, source_id, camera_intrinsics)
            color_dst, depth_dst = load_images(path, target_id, camera_intrinsics)

            trans_s_to_t = marker_registration(color_src, color_dst)

            if trans_s_to_t is None:
                if target_id != source_id + 1:
                    continue
                else:
                    print('ERROR! The number of aruco codes that can be matched by adjacent frames is insufficient, '
                          'please re-record')  # TODO ICP?
                    return
            else:
                if target_id == source_id + 1:
                    odometry = np.dot(odometry, trans_s_to_t)
                    pose_graph.add_vertex(target_id, g2o.Isometry3d(odometry))
                    pose_graph.add_edge(vertices=(source_id, target_id), measurement=g2o.Isometry3d(trans_s_to_t))
                else:
                    assert target_id > source_id
                    pose_graph.add_edge(vertices=(source_id, target_id),
                                        measurement=g2o.Isometry3d(trans_s_to_t))  # TODO rubust kernel

    return pose_graph


def load_images(path, ID, camera_intrinsics, ):
    """
    Load a color and a depth image by path and image ID

    """

    img_file = os.path.join(path, 'rgb', f'{(ID * LABEL_INTERVAL):06}' + '.png')
    cad = cv2.imread(img_file)

    depth_file = os.path.join(path, 'depth', f'{(ID * LABEL_INTERVAL):06}' + '.png')
    reader = png.Reader(depth_file)
    pngdata = reader.read()
    # depth = np.vstack(map(np.uint16, pngdata[2]))
    depth = np.array(tuple(map(np.uint16, pngdata[2])))
    pointcloud = convert_depth_frame_to_pointcloud(depth, camera_intrinsics)

    return (cad, pointcloud)


def load_pcd(path, Filename, camera_intrinsics, downsample=True, interval=1):
    """
     load pointcloud by path and down samle (if True) based on voxel_size

     """

    global voxel_size

    img_file = os.path.join(path, 'rgb', f'{(Filename * interval):06}' + '.png')

    cad = cv2.imread(img_file)
    cad = cv2.cvtColor(cad, cv2.COLOR_BGR2RGB)
    depth_file = os.path.join(path, 'depth', f'{(Filename * interval):06}' + '.png')
    reader = png.Reader(depth_file)
    pngdata = reader.read()
    # depth = np.vstack(map(np.uint16, pngdata[2]))
    depth = np.array(tuple(map(np.uint16, pngdata[2])))
    mask = depth.copy()
    depth = convert_depth_frame_to_pointcloud(depth, camera_intrinsics)

    source = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(depth[mask > 0])
    source.colors = open3d.utility.Vector3dVector(cad[mask > 0])

    if downsample == True:
        source = source.voxel_down_sample(voxel_size=voxel_size)
        source.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=0.002 * 2, max_nn=30))

    return source


def compute_camera_pose(folder_path, camera_intrinsics):
    global mtx
    mtx = np.array([[camera_intrinsics['fx'], 0., camera_intrinsics['ppx']],
                    [0., camera_intrinsics['fy'], camera_intrinsics['ppy']],
                    [0., 0., 1.]])
    Ts = []
    T_npy_path = os.path.join(folder_path, 'transform.npy')
    if os.path.exists(T_npy_path):
        print('transform file exist')
        return np.load(T_npy_path)
    else:
        n_pcds = int(len(glob.glob1(folder_path + "/rgb", "*.png")) / LABEL_INTERVAL)
        print("Full registration ...")
        pose_graph = full_registration(folder_path, camera_intrinsics, n_pcds)

        print("Optimizing PoseGraph ...")
        pose_graph.optimize(1000)

        num_annotations = int(len(glob.glob1(folder_path + "/rgb", "*.png")) / LABEL_INTERVAL)

        for point_id in range(num_annotations):
            Ts.append(poseRt(pose_graph.get_pose(point_id).R, pose_graph.get_pose(point_id).t))
        Ts = np.array(Ts)
        np.save(T_npy_path, Ts)
        print("Transforms saved")
        return Ts
