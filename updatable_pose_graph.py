# The pose graph in open3d is not modifiable, so we implemented a modifiable pose graph.
from open3d import pipelines

from params import max_correspondence_distance_fine


class PoseGraph:
    def __init__(self):
        self.Node = {}
        self.Edge = {}

    def Set_Node(self, node_id, T_obj):
        self.Node[node_id] = T_obj

    def Set_Edge(self, Source_id, Target_id, T_obj):
        tag = str(Source_id) + '_' + str(Target_id)
        self.Edge[tag] = T_obj

    def sort_node(self):
        dic_sorted = sorted(self.Node.items())
        self.Node = {k: v for k, v in dic_sorted}

    def optimize(self):
        pose_graph = pipelines.registration.PoseGraph()
        self.sort_node()

        for node in self.Node.values():
            pose_graph.nodes.append(node)
        for edge in self.Edge.values():
            pose_graph.edges.append(edge)

        option = pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0)
        pipelines.registration.global_optimization(pose_graph,
                                                   pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                                   pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                   option)
        self.Node.clear()
        self.Edge.clear()

        for i, node in enumerate(pose_graph.nodes):
            self.Set_Node(i, node)
        for edge in pose_graph.edges:
            self.Set_Edge(edge.source_node_id, edge.target_node_id, edge)

    def get_pose(self, node_id):
        return self.Node[node_id].pose

    def get_Edge(self, Source_id, Target_id):
        tag = str(Source_id) + '_' + str(Target_id)
        try:
            return self.Edge[tag]
        except:
            return None

    def get_len(self):
        return len(self.Node)
