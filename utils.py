"""
registration.py
---------------

Functions for registering (aligning) point clouds with ICP and feature registration.

"""

from open3d import *
import numpy as np
import cv2
from open3d import pipelines
import png
from params import VOXEL_SIZE, LABEL_INTERVAL, N_Neighbours
from joblib import Parallel, delayed


def icp(source, target, voxel_size, max_correspondence_distance_coarse, max_correspondence_distance_fine,
        method="colored-icp"):
    """
    Perform pointcloud registration using iterative closest point.

    Parameters
    ----------
    source : An open3d.Pointcloud instance
      6D pontcloud of a source segment
    target : An open3d.Pointcloud instance
      6D pointcloud of a target segment
    method : string
      colored-icp, as in Park, Q.-Y. Zhou, and V. Koltun, Colored Point Cloud
      Registration Revisited, ICCV, 2017 (slower)
      point-to-plane, a coarse to fine implementation of point-to-plane icp (faster)
    max_correspondence_distance_coarse : float
      The max correspondence distance used for the course ICP during the process
      of coarse to fine registration (if point-to-plane)
    max_correspondence_distance_fine : float
      The max correspondence distance used for the fine ICP during the process
      of coarse to fine registration (if point-to-plane)

    Returns
    ----------
    transformation_icp: (4,4) float
      The homogeneous rigid transformation that transforms source to the target's
      frame
    information_icp:
      An information matrix returned by open3d.get_information_matrix_from_ \
      point_clouds function
    """

    assert method in ["point-to-plane", "colored-icp"], "point-to-plane or colored-icp"
    if method == "point-to-plane":
        icp_coarse = pipelines.registration.registration_icp(source, target,
                                                             max_correspondence_distance_coarse, np.identity(4),
                                                             pipelines.registration.TransformationEstimationPointToPlane())
        icp_fine = pipelines.registration.registration_icp(source, target,
                                                           max_correspondence_distance_fine, icp_coarse.transformation,
                                                           pipelines.registration.TransformationEstimationPointToPlane())

        transformation_icp = icp_fine.transformation

    if method == "colored-icp":
        result_icp = pipelines.registration.registration_colored_icp(source, target, voxel_size, np.identity(4),
                                                                     pipelines.registration.ICPConvergenceCriteria(
                                                                         relative_fitness=1e-8,
                                                                         relative_rmse=1e-8, max_iteration=50))

        transformation_icp = result_icp.transformation

    information_icp = pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        transformation_icp)

    return transformation_icp, information_icp


def feature_registration(source, target, MIN_MATCH_COUNT=12):
    """
    Obtain the rigid transformation from source to target
    first find correspondence of color images by performing fast registration
    using SIFT features on color images.
    The corresponding depth values of the matching keypoints is then used to
    obtain rigid transformation through a ransac process.


    Parameters
    ----------
    source : ((n,m) uint8, (n,m) float)
      The source color image and the corresponding 3d pointcloud combined in a list
    target : ((n,m) uint8, (n,m) float)
      The target color image and the corresponding 3d pointcloud combined in a list
    MIN_MATCH_COUNT : int
      The minimum number of good corresponding feature points for the algorithm  to
      trust the pairwise registration result with feature matching only

    Returns
    ----------
    transform: (4,4) float or None
      The homogeneous rigid transformation that transforms source to the target's
      frame
      if None, registration result using feature matching only cannot be trusted
      either due to no enough good matching feature points are found, or the ransac
      process does not return a solution

    """
    cad_src, depth_src = source
    cad_des, depth_des = target

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descripto  rs with SIFT
    kp1, des1 = sift.detectAndCompute(cad_src, None)
    kp2, des2 = sift.detectAndCompute(cad_des, None)

    # find good mathces
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)

    # if number of good matching feature point is greater than the MIN_MATCH_COUNT

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        bad_match_index = np.where(np.array(matchesMask) == 0)

        src_index = np.vstack(src_pts).squeeze()
        src_index = np.delete(src_index, tuple(bad_match_index[0]), axis=0)
        src_index[:, [0, 1]] = src_index[:, [1, 0]]
        src_index = tuple(src_index.T.astype(np.int32))
        src_depths = depth_src[src_index]

        dst_index = np.vstack(dst_pts).squeeze()
        dst_index = np.delete(dst_index, tuple(bad_match_index[0]), axis=0)
        dst_index[:, [0, 1]] = dst_index[:, [1, 0]]
        dst_index = tuple(dst_index.T.astype(np.int32))
        dst_depths = depth_des[dst_index]

        dst_good = []
        src_good = []

        for i in range(len(dst_depths)):
            if np.sum(dst_depths[i]) != 0 and np.sum(src_depths[i]) != 0:
                dst_good.append(dst_depths[i])
                src_good.append(src_depths[i])

        return src_good, dst_good

    else:
        return [], []


def ransac_iteration(p, p_prime, threshold, n, k):
    idx = np.random.choice(n, k, replace=False)
    R_temp, t_temp = rigid_transform_3D(p[idx, :], p_prime[idx, :])
    R_temp = np.array(R_temp)
    t_temp = np.array(t_temp).T[0]
    # Error
    transformed = np.dot(R_temp, p.T).T + t_temp
    error = (transformed - p_prime) ** 2
    error = np.sum(error, axis=1)
    error = np.sqrt(error)

    inliers = np.where(error < threshold)[0]
    num_inliers = len(inliers)
    return inliers, num_inliers


def match_ransac(p, p_prime, num_iterations=3000, tol=0.005):
    """
    A ransac process that estimates the transform between two set of points
    p and p_prime.
    The transform is returned if the RMSE of the smallest 70% is smaller
    than the tol.

    Parameters
    ----------
    p : (n,3) float
      The source 3d pointcloud as a numpy.ndarray
    target : (n,3) float
      The target 3d pointcloud as a numpy.ndarray
    tol : float
      A transform is considered found if the smallest 70% RMSE error between the
      transformed p to p_prime is smaller than the tol

    Returns
    ----------
    transform: (4,4) float or None
      The homogeneous rigid transformation that transforms p to the p_prime's
      frame
      if None, the ransac does not find a sufficiently good solution

    """
    k = 18
    assert len(p) == len(p_prime)
    max_inliers = 0
    n = len(p)
    for i in range(num_iterations):
        idx = np.random.choice(n, k, replace=False)
        R_temp, t_temp = rigid_transform_3D(p[idx, :], p_prime[idx, :])
        R_temp = np.array(R_temp)
        t_temp = np.array(t_temp).T[0]
        # Error
        transformed = np.dot(R_temp, p.T).T + t_temp
        error = (transformed - p_prime) ** 2
        error = np.sum(error, axis=1)
        error = np.sqrt(error)

        inliers = np.where(error < tol)[0]
        num_inliers = len(inliers)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_R, best_t = rigid_transform_3D(p[inliers, :], p_prime[inliers, :])
            best_R = np.array(best_R)
            best_t = np.array(best_t).T[0]
    #
    # results = Parallel(n_jobs=-1)(delayed(ransac_iteration)(p, p_prime, tol, n, k) for i in range(num_iterations))
    # for inliers, num_inliers in results:
    #     if num_inliers > max_inliers:
    #         max_inliers = num_inliers
    #         best_R, best_t = rigid_transform_3D(p[inliers, :], p_prime[inliers, :])
    #         best_R = np.array(best_R)
    #         best_t = np.array(best_t).T[0]
    #
    transformed = np.dot(best_R, p.T).T + best_t
    error = (transformed - p_prime) ** 2
    error = np.sum(error, axis=1)
    error = np.sqrt(error)
    inliers = np.where(error < tol)[0]
    best_R, best_t = rigid_transform_3D(p[inliers, :], p_prime[inliers, :])
    R = np.array(best_R)
    tt = np.array(best_t).T[0]
    transform = [[R[0][0], R[0][1], R[0][2], tt[0]],
                 [R[1][0], R[1][1], R[1][2], tt[1]],
                 [R[2][0], R[2][1], R[2][2], tt[2]],
                 [0, 0, 0, 1]]
    return transform


def rigid_transform_3D(A, B):
    """
    Estimate a rigid transform between 2 set of points of equal length
    through singular value decomposition(svd), return a rotation and a
    transformation matrix

    Parameters
    ----------
    A : (n,3) float
      The source 3d pointcloud as a numpy.ndarray
    B : (n,3) float
      The target 3d pointcloud as a numpy.ndarray

    Returns
    ----------
    R: (3,3) float
      A rigid rotation matrix
    t: (3) float
      A translation vector

    """

    assert len(A) == len(B)
    A = np.asmatrix(A)
    B = np.asmatrix(B)
    N = A.shape[0]

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    H = AA.T * BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T

    # reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A.T + centroid_B.T

    return (R, t)


def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret


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
        source = source.voxel_down_sample(voxel_size=VOXEL_SIZE)
        source.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=0.002 * 2, max_nn=30))

    return source


def make_target_frame_list(source_id, n_pcds):
    target_frame_list = list(range(source_id + 1, n_pcds, max(1, int(n_pcds / N_Neighbours))))
    for i in range(N_Neighbours):
        target_frame_list.append(min(n_pcds - 1, source_id + 3 * i))
    target_frame_list = list(set(target_frame_list))
    target_frame_list.sort()
    return target_frame_list
