import g2o
import numpy as np

from BA_Optimization.BA_manager import BA_Manager
from BA_Optimization.components import Measurement, RGBDFrame, Camera
from BA_Optimization.covisibility import CovisibilityGraph
from BA_Optimization.img_feature import ImageFeature
from Front_Odometer import Front_Odometer
from params import BA_param


class Camera_Track(object):
    def __init__(self, pose_graph):

        self.graph = CovisibilityGraph()
        self.pose_graph = pose_graph
        self.ba_manager = BA_Manager(self.graph, self.pose_graph)
        self.front_odom = Front_Odometer(self.pose_graph)

        self.reference = None  # reference keyframe
        self.preceding = None  # last keyframe
        self.current = None  # current frame
        self.candidates = []  # candidate keyframes
        self.results = []  # tracking results

        self.cam = Camera(
            BA_param.cam_fx, BA_param.cam_fy, BA_param.cam_cx, BA_param.cam_cy,
            BA_param.cam_width, BA_param.cam_height, BA_param.cam_scale,
            BA_param.cam_virtual_baseline, BA_param.cam_depth_near, BA_param.cam_depth_far,
            BA_param.cam_frustum_near, BA_param.cam_frustum_far)

    def initialize(self, index, rgb_image, depth_cloudify, depth, pcd):
        if BA_param.use_BA:
            feature = ImageFeature(rgb_image)
            feature.extract()
            frame = RGBDFrame(index, g2o.Isometry3d(), feature, depth, self.cam)

            mappoints, measurements = frame.cloudify()

            keyframe = frame.to_keyframe()
            keyframe.set_fixed(True)
            self.graph.add_keyframe(keyframe)
            self.ba_manager.add_measurements(keyframe, mappoints, measurements)

            self.reference = keyframe
            self.preceding = keyframe
            self.current = keyframe

        self.front_odom.initialize(rgb_image, depth_cloudify, pcd)

    def track(self, index, rgb_image, depth_cloudify, depth, pcd):
        if BA_param.use_BA:
            feature = ImageFeature(rgb_image)
            feature.extract()
            frame = RGBDFrame(index, g2o.Isometry3d(), feature, depth, self.cam)

            self.current = frame

        predicted_pose = self.front_odom.track(rgb_image, depth_cloudify, pcd, index)
        if BA_param.use_BA:
            predicted_pose = g2o.Isometry3d(predicted_pose)
            frame.update_pose(predicted_pose)

            local_mappoints = self.filter_points(frame)
            measurements = frame.match_mappoints(
                local_mappoints, Measurement.Source.TRACKING)

            tracked_map = set()
            for m in measurements:
                mappoint = m.mappoint
                mappoint.update_descriptor(m.get_descriptor())
                mappoint.increase_measurement_count()
                tracked_map.add(mappoint)

            try:
                self.reference = self.graph.get_reference_frame(tracked_map)
                self.candidates.append(frame)
                self.results.append(True)  # tracking succeed
            except:
                self.results.append(False)  # tracking fail

            remedy = False
            if self.results[-2:].count(False) == 2:  # Useless
                if (len(self.candidates) > 0 and
                        self.candidates[-1].idx > self.preceding.idx):
                    frame = self.candidates[-1]
                    remedy = True

            if remedy or self.results[-1]:
                keyframe = frame.to_keyframe()
                keyframe.update_reference(self.reference)
                keyframe.update_preceding(self.preceding)

                self.ba_manager.add_keyframe(keyframe, measurements)
                self.preceding = keyframe

    def filter_points(self, frame):
        local_mappoints = self.graph.get_local_map_v2(
            [self.preceding, self.reference])[0]

        can_view = frame.can_view(local_mappoints)

        checked = set()
        filtered = []
        for i in np.where(can_view)[0]:
            pt = local_mappoints[i]
            if pt.is_bad():
                continue
            pt.increase_projection_count()
            filtered.append(pt)
            checked.add(pt)

        for reference in {self.preceding, self.reference}:
            for pt in reference.mappoints():  # neglect can_view test
                if pt in checked or pt.is_bad():
                    continue
                pt.increase_projection_count()
                filtered.append(pt)

        return filtered
