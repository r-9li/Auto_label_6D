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
from params import VOXEL_SIZE, LABEL_INTERVAL, N_Neighbours, invisible_detect_voxel_size, invisible_detect_threshold, \
    BA_param
from joblib import Parallel, delayed
import open3d as o3d
import json
import cv2
import cv2.aruco as aruco


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


def multiscale_icp(source, target, voxel_size, max_iter, method_list, init_transformation=np.identity(4)):
    """
    transformation_icp: (4,4) float
      The homogeneous rigid transformation that transforms source to the target's
      frame
    information_icp:
      An information matrix returned by open3d.get_information_matrix_from_ \
      point_clouds function
    """

    assert len(voxel_size) == len(max_iter) == len(method_list)
    current_transformation = init_transformation

    for i in range(len(voxel_size)):
        method = method_list[i]
        assert method in ["point_to_plane", "color", "generalized"], "point_to_plane or color or generalized"
        iter = max_iter[i]
        distance_threshold = voxel_size[i] * 1.4
        source_down = source.voxel_down_sample(voxel_size[i])
        target_down = target.voxel_down_sample(voxel_size[i])

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[i] * 2.0, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[i] * 2.0, max_nn=30))

        if method == "point_to_plane":
            loss = o3d.pipelines.registration.TukeyLoss(0.1)
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down, target_down, distance_threshold,
                current_transformation,
                o3d.pipelines.registration.
                TransformationEstimationPointToPlane(loss),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=iter))
        elif method == "color":
            # Colored ICP is sensitive to threshold.
            # Fallback to preset distance threshold that works better.
            result_icp = o3d.pipelines.registration.registration_colored_icp(
                source_down, target_down, voxel_size[i],
                current_transformation,
                o3d.pipelines.registration.
                TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=iter))
        elif method == "generalized":
            result_icp = o3d.pipelines.registration.registration_generalized_icp(
                source_down, target_down, distance_threshold,
                current_transformation,
                o3d.pipelines.registration.
                TransformationEstimationForGeneralizedICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=iter))
        current_transformation = result_icp.transformation
        if i == len(max_iter) - 1:
            information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                source_down, target_down, voxel_size[i] * 1.4,
                result_icp.transformation)

    return result_icp.transformation, information_matrix


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

    results = Parallel(n_jobs=-1)(delayed(ransac_iteration)(p, p_prime, tol, n, k) for i in range(num_iterations))
    for inliers, num_inliers in results:
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers

    best_R, best_t = rigid_transform_3D(p[best_inliers, :], p_prime[best_inliers, :])
    best_R = np.array(best_R)
    best_t = np.array(best_t).T[0]

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

    return cad, pointcloud, depth


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
    if not BA_param.use_BA:
        for i in range(N_Neighbours):
            target_frame_list.append(min(n_pcds - 1, source_id + 3 * i))
        target_frame_list = list(set(target_frame_list))
        target_frame_list.sort()

    return target_frame_list


def _make_point_cloud(rgb_img, depth_img, cam_K):
    # convert images to open3d types
    rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    depth_img_o3d = o3d.geometry.Image(depth_img)

    # convert image to point cloud
    intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb_img.shape[0], rgb_img.shape[1],
                                                  cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, depth_img_o3d,
                                                              depth_scale=1, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    return pcd


def _detect_invisible_object_iter(current_image_index, scene_path, gt_6d_pose_data, objects_path):
    return_list = []
    camera_params_path = os.path.join(scene_path, 'scene_camera.json')
    with open(camera_params_path) as f:
        data = json.load(f)
        cam_K = data[str(current_image_index)]['cam_K']
        cam_K = np.array(cam_K).reshape((3, 3))
        depth_scale = data[str(current_image_index)]['depth_scale']

    rgb_path = os.path.join(scene_path, 'rgb', f'{current_image_index:06}' + '.png')
    rgb_img = cv2.imread(rgb_path)
    depth_path = os.path.join(scene_path, 'depth', f'{current_image_index:06}' + '.png')
    depth_img = cv2.imread(depth_path, -1)
    depth_img = np.float32(depth_img * depth_scale / 1000)
    try:
        geometry = _make_point_cloud(rgb_img, depth_img, cam_K)  # scene point cloud
        if not geometry.has_normals():
            geometry.estimate_normals()
        geometry.normalize_normals()
    except Exception:
        print("Failed to generate scene point cloud.")

    try:
        scene_data = gt_6d_pose_data[str(current_image_index)]
        for obj in scene_data:
            obj_geometry = o3d.io.read_point_cloud(
                os.path.join(objects_path, 'obj_' + f"{int(obj['obj_id']):06}" + '.ply'))
            obj_geometry.points = o3d.utility.Vector3dVector(
                np.array(obj_geometry.points) / 1000)  # convert mm to meter

            translation = np.array(np.array(obj['cam_t_m2c']), dtype=np.float64) / 1000  # convert to meter
            orientation = np.array(np.array(obj['cam_R_m2c']), dtype=np.float64)
            transform = np.concatenate((orientation.reshape((3, 3)), translation.reshape(3, 1)), axis=1)
            transform_cam_to_obj = np.concatenate(
                (transform, np.array([0, 0, 0, 1]).reshape(1, 4)))  # homogeneous transform

            obj_geometry.transform(transform_cam_to_obj)

            obj_geometry_obb = obj_geometry.get_oriented_bounding_box()
            crop_geometry = geometry.crop(obj_geometry_obb)
            if len(crop_geometry.points) == 0:
                return_list.append(0)
                del obj_geometry, crop_geometry
                continue
            voxel_size = invisible_detect_voxel_size
            obj_geometry_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(input=obj_geometry,
                                                                                voxel_size=voxel_size)
            crop_geometry_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(input=crop_geometry,
                                                                                 voxel_size=voxel_size)
            obj_geometry_voxel_index = obj_geometry_voxel.get_voxels()
            crop_geometry_voxel_index = crop_geometry_voxel.get_voxels()
            obj_geometry_downsample = []
            for voxel in obj_geometry_voxel_index:
                obj_geometry_downsample.append(
                    obj_geometry_voxel.get_voxel_center_coordinate(voxel.grid_index))
            obj_geometry_downsample = np.array(obj_geometry_downsample)
            crop_geometry_downsample = []
            for voxel in crop_geometry_voxel_index:
                crop_geometry_downsample.append(
                    crop_geometry_voxel.get_voxel_center_coordinate(voxel.grid_index))
            crop_geometry_downsample = np.array(crop_geometry_downsample)

            crop_geometry_downsample_pcd = o3d.geometry.PointCloud()
            crop_geometry_downsample_pcd.points = o3d.utility.Vector3dVector(crop_geometry_downsample)
            crop_geometry_downsample_pcd_tree = o3d.geometry.KDTreeFlann(crop_geometry_downsample_pcd)
            distances = []
            for point in obj_geometry_downsample:
                _, _, point_dist = crop_geometry_downsample_pcd_tree.search_knn_vector_3d(point, 1)
                distances.append(point_dist)
            distances = np.array(distances)
            overlap_ratio = np.mean(
                distances < voxel_size)  # overlap_ratio=overlap_point/obj_point

            if overlap_ratio < invisible_detect_threshold:  # threshold
                return_list.append(0)  # zero means remove
            else:
                return_list.append(1)

            del obj_geometry, obj_geometry_voxel, obj_geometry_voxel_index, obj_geometry_downsample
            del crop_geometry, crop_geometry_voxel, crop_geometry_voxel_index, crop_geometry_downsample, crop_geometry_downsample_pcd, crop_geometry_downsample_pcd_tree

        del geometry
    except Exception as e:
        print(e)
    return current_image_index, return_list
