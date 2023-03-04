VOXEL_SIZE = 0.001
max_correspondence_distance_coarse = VOXEL_SIZE * 15
max_correspondence_distance_fine = VOXEL_SIZE * 1.5
LABEL_INTERVAL = 1
N_Neighbours = 12
ICP_METHOD = "point-to-plane"
RECORD_FRAME_NUM = 2400
Interval_frame = 3
object_icp_voxel_list = [0.004, 0.0033, 0.0027, 0.002, 0.0015, 0.001]
object_icp_iter_list = [900, 900, 900, 900, 900, 900]
object_icp_method_list = ["generalized", "generalized", "generalized", "point_to_plane", "point_to_plane",
                          "point_to_plane"]
