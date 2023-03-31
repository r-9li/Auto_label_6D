import cv2

VOXEL_SIZE = 0.001
max_correspondence_distance_coarse = VOXEL_SIZE * 15
max_correspondence_distance_fine = VOXEL_SIZE * 1.5
LABEL_INTERVAL = 1
N_Neighbours = 18
ICP_METHOD = "point-to-plane"
RECORD_FRAME_NUM = 2400
Interval_frame = 3
object_icp_voxel_list = [0.004, 0.0033, 0.0027, 0.002, 0.0015, 0.001]
object_icp_iter_list = [900, 900, 900, 900, 900, 900]
object_icp_method_list = ["generalized", "generalized", "generalized", "point_to_plane", "point_to_plane",
                          "point_to_plane"]
invisible_detect_voxel_size = 0.0006
invisible_detect_threshold = 0.48


class BA_param:
    # Feature
    extractor = cv2.ORB_create()

    descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matching_distance = 30
    matching_cell_size = 6  # pixels
    matching_neighborhood = 1

    local_window_size = 15
    ba_max_iterations = 90
    global_ba_max_iterations = 120
    # TODO Get information from json file

    cam_virtual_baseline = 0.0497251264750957  # meter
    cam_depth_near = 0.15
    cam_depth_far = 10.0
    cam_frustum_near = 0.1
    cam_frustum_far = 50.0
    cam_fx = 604.182189941406
    cam_fy = 603.487915039062
    cam_cx = 323.661773681641
    cam_cy = 248.717361450195
    cam_width = 640
    cam_height = 480
    cam_scale = 1000  # TODO The depth_scale used here is defined like this: depth / depth_scale=meter, but it is
    # TODO defined like this elsewhere: depth*depth_scale=meter(mm).
    use_BA = False
