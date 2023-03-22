from itertools import chain

import numpy as np
from open3d import pipelines
from BA_Optimization.components import Measurement
from BA_Optimization.optimization import LocalBA
from params import BA_param


class BA_Manager(object):
    def __init__(self, graph, pose_graph):
        self.graph = graph
        self.pose_graph = pose_graph
        self.local_keyframes = []

        self.optimizer = LocalBA()

    def add_keyframe(self, keyframe, measurements):  # Local BA
        self.graph.add_keyframe(keyframe)
        self.create_points(keyframe)

        for m in measurements:
            self.graph.add_measurement(keyframe, m.mappoint, m)

        self.local_keyframes.clear()
        self.local_keyframes.append(keyframe)

        self.fill(self.local_keyframes, keyframe)
        self.refind(self.local_keyframes, self.get_owned_points(keyframe))

        self.bundle_adjust(self.local_keyframes)
        self.points_culling(self.local_keyframes)
        self.update_PoseGraph()

    def fill(self, keyframes, keyframe):
        covisible = sorted(
            keyframe.covisibility_keyframes().items(),
            key=lambda _: _[1], reverse=True)

        for kf, n in covisible:
            if n > 0 and kf not in keyframes and self.is_safe():
                keyframes.append(kf)
                if len(keyframes) >= BA_param.local_window_size:
                    return

    def create_points(self, keyframe):
        mappoints, measurements = keyframe.cloudify()
        self.add_measurements(keyframe, mappoints, measurements)

    def add_measurements(self, keyframe, mappoints, measurements):
        for mappoint, measurement in zip(mappoints, measurements):
            self.graph.add_mappoint(mappoint)
            self.graph.add_measurement(keyframe, mappoint, measurement)
            mappoint.increase_measurement_count()

    def bundle_adjust(self, keyframes):
        adjust_keyframes = set()
        for kf in keyframes:
            if not kf.is_fixed():
                adjust_keyframes.add(kf)

        fixed_keyframes = set()
        for kf in adjust_keyframes:
            for ck, n in kf.covisibility_keyframes().items():
                if (n > 0 and ck not in adjust_keyframes
                        and self.is_safe() and ck < kf):
                    fixed_keyframes.add(ck)

        self.optimizer.set_data(adjust_keyframes, fixed_keyframes)
        completed = self.optimizer.optimize(BA_param.ba_max_iterations)

        self.optimizer.update_poses()
        self.optimizer.update_points()

        if completed:
            self.remove_measurements(self.optimizer.get_bad_measurements())
        return completed

    def is_safe(self):
        return True

    def get_owned_points(self, keyframe):
        owned = []
        for m in keyframe.measurements():
            if m.from_triangulation():
                owned.append(m.mappoint)
        return owned

    def filter_unmatched_points(self, keyframe, mappoints):
        filtered = []
        for i in np.where(keyframe.can_view(mappoints))[0]:
            pt = mappoints[i]
            if (not pt.is_bad() and
                    not self.graph.has_measurement(keyframe, pt)):
                filtered.append(pt)
        return filtered

    def refind(self, keyframes, new_mappoints):  # time consuming
        if len(new_mappoints) == 0:
            return
        for keyframe in keyframes:
            filtered = self.filter_unmatched_points(keyframe, new_mappoints)
            if len(filtered) == 0:
                continue
            for mappoint in filtered:
                mappoint.increase_projection_count()

            measuremets = keyframe.match_mappoints(filtered, Measurement.Source.REFIND)

            for m in measuremets:
                self.graph.add_measurement(keyframe, m.mappoint, m)
                m.mappoint.increase_measurement_count()

    def remove_measurements(self, measurements):
        for m in measurements:
            m.mappoint.increase_outlier_count()
            self.graph.remove_measurement(m)

    def points_culling(self, keyframes):  # Remove bad mappoints
        mappoints = set(chain(*[kf.mappoints() for kf in keyframes]))
        for pt in mappoints:
            if pt.is_bad():
                self.graph.remove_mappoint(pt)

    def update_PoseGraph(self):
        node_change_list = []
        for kf in self.local_keyframes:
            kf_id = kf.idx
            node_change_list.append(kf_id)

            R = kf.pose.R
            t = kf.pose.t
            T = np.concatenate((R, t.reshape(3, 1)), axis=1)
            T = np.concatenate(
                (T, np.array([0, 0, 0, 1]).reshape(1, 4)))
            self.pose_graph.Set_Node(kf_id, pipelines.registration.PoseGraphNode(np.linalg.inv(T)))

        node_change_list.sort()
        for node_id in node_change_list:
            if node_id == 0:  # The first frame
                back_edge = self.pose_graph.get_Edge(node_id, node_id + 1)
                back_node_pose = self.pose_graph.get_pose(node_id + 1)

                current_node_pose = self.pose_graph.get_pose(node_id)

                back_T = np.dot(back_node_pose, np.linalg.inv(current_node_pose))

                self.pose_graph.Set_Edge(node_id, node_id + 1,
                                         pipelines.registration.PoseGraphEdge(node_id, node_id + 1,
                                                                              back_T,
                                                                              back_edge.information,
                                                                              uncertain=False))
            elif node_id == self.pose_graph.get_len() - 1:  # The last frame
                front_edge = self.pose_graph.get_Edge(node_id - 1, node_id)
                front_node_pose = self.pose_graph.get_pose(node_id - 1)

                current_node_pose = self.pose_graph.get_pose(node_id)

                front_T = np.dot(current_node_pose, np.linalg.inv(front_node_pose))

                self.pose_graph.Set_Edge(node_id - 1, node_id,
                                         pipelines.registration.PoseGraphEdge(node_id - 1, node_id,
                                                                              front_T,
                                                                              front_edge.information,
                                                                              uncertain=False))
            else:  # Normal frame
                front_edge = self.pose_graph.get_Edge(node_id - 1, node_id)
                front_node_pose = self.pose_graph.get_pose(node_id - 1)

                back_edge = self.pose_graph.get_Edge(node_id, node_id + 1)
                back_node_pose = self.pose_graph.get_pose(node_id + 1)

                current_node_pose = self.pose_graph.get_pose(node_id)

                front_T = np.dot(current_node_pose, np.linalg.inv(front_node_pose))
                back_T = np.dot(back_node_pose, np.linalg.inv(current_node_pose))

                self.pose_graph.Set_Edge(node_id - 1, node_id,
                                         pipelines.registration.PoseGraphEdge(node_id - 1, node_id,
                                                                              front_T,
                                                                              front_edge.information,
                                                                              uncertain=False))

                self.pose_graph.Set_Edge(node_id, node_id + 1,
                                         pipelines.registration.PoseGraphEdge(node_id, node_id + 1,
                                                                              back_T,
                                                                              back_edge.information,
                                                                              uncertain=False))
