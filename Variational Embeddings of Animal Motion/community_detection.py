import numpy as np
from pyvis.network import Network
import networkx as nx
from scipy.stats.contingency import crosstab
from typing import Optional, List, Tuple


class CommunityDetection(object):

    def __init__(self,
                 n_clusters: int = 8,
                 tree_cut: int = 2,
                 threshold: float = 0.):
        self.tree_cut = tree_cut
        self.n_clusters = n_clusters
        self.threshold = threshold

        self.tree_ = None
        self.transition_matrix_ = None
        self.motif_probabilities_ = None
        self.communities_ = None

        self._root_node = "Root"

    def fit(self, labels: np.ndarray) -> List[List[str]]:
        _, motif_frequencies = np.unique(labels, return_counts=True)
        self.motif_probabilities_ = motif_frequencies / motif_frequencies.sum(axis=0)

        # merge transition_matrix
        adjacency_matrix = self.build_adjacency_matrix(labels)
        self.transition_matrix_ = self.build_transition_matrix(adjacency_matrix)

        # build Tree.
        self.tree_ = self.graph_to_tree(self.transition_matrix_, self.motif_probabilities_)

        # build communities from tree
        self.communities_ = self.build_communities(self._root_node, [])

        # update the color of tree and edges
        self._update_tree_groups(self._root_node)

        return self.communities_

    def _update_tree_groups(self, node: str, group: Optional[int] = None) -> None:
        def get_community_index(node):
            if node.startswith("h"):  # if parent
                node = node.split("_")[1]

            for i, community in enumerate(self.communities_):
                if node in community:
                    return i + 1  # as group 1 is reserved for non-grouped nodes
            return None

        if self.tree_ is None:
            raise ValueError("Please create a tree first.")

        if self.tree_.nodes[node]["level"] >= self.tree_cut:
            if group is None:
                group = get_community_index(node)
            self.tree_.nodes[node]["group"] = group
        else:
            self.tree_.nodes[node]["color"] = "grey"
            self.tree_.nodes[node]["group"] = 0

        children = list(self.tree_.successors(node))
        if len(children) > 0:
            for child in children:
                self._update_tree_groups(child, group=group)

    def build_adjacency_matrix(self, labels: np.ndarray) -> np.ndarray:
        cluster_range = np.arange(self.n_clusters)
        _, count = crosstab(labels[:-1], labels[1:], levels=(cluster_range, cluster_range))
        np.fill_diagonal(count, 0.)
        return count

    def build_transition_matrix(self, adjacency_matrix) -> np.ndarray:
        transition_matrix = adjacency_matrix / adjacency_matrix.sum(axis=1)[:, np.newaxis]
        transition_matrix[transition_matrix <= self.threshold] = 0.
        np.nan_to_num(transition_matrix, nan=0.)
        return transition_matrix

    def graph_to_tree(self, transition_matrix, motif_probabilities) -> nx.DiGraph:
        modified_transition_matrix = transition_matrix.copy()
        motif_norm_temp = motif_probabilities.copy()
        n_clusters = len(motif_probabilities)

        merge_nodes = []
        cluster_levels = np.zeros((n_clusters,))

        while modified_transition_matrix.sum() != 0:
            left, right = self.merge_nodes_cost(modified_transition_matrix, motif_norm_temp)

            merge_nodes.append((left, right))
            cluster_levels[left] += 1
            cluster_levels[right] += 1

            merged_usage = motif_norm_temp[left] + motif_norm_temp[right]
            merged_transition_y = modified_transition_matrix[:, left] + modified_transition_matrix[:, right]
            merged_transition_x = modified_transition_matrix[left, :] + modified_transition_matrix[right, :]

            # set right
            motif_norm_temp[right] = merged_usage
            modified_transition_matrix[:, right] = merged_transition_y
            modified_transition_matrix[right, :] = merged_transition_x
            modified_transition_matrix[right, right] = 0

            # clear left
            motif_norm_temp[left] = 0
            modified_transition_matrix[:, left] = 0
            modified_transition_matrix[left, :] = 0

        T = nx.DiGraph(level=0, group=0)
        T.add_node(self._root_node, level=0, group=0)

        node_dict = dict()

        for i, (left, right) in enumerate(merge_nodes[::-1]):
            if right in node_dict:
                parent = node_dict[right]
                level = int(parent.split("_")[-1]) + 1
            else:
                parent = self._root_node
                level = 1

            right_edge_name = f"h_{right}_{level}" if cluster_levels[right] > 1 else str(right)
            left_edge_name = f"h_{left}_{level}" if cluster_levels[left] > 1 else str(left)

            T.add_node(right_edge_name, level=level, group=0)
            T.add_node(left_edge_name, level=level, group=0)
            T.add_edge(parent, right_edge_name)
            T.add_edge(parent, left_edge_name)

            node_dict.update({right: right_edge_name, left: left_edge_name})

            cluster_levels[left] -= 1
            cluster_levels[right] -= 1

        return T

    def merge_nodes_cost(self, transition_matrix: np.ndarray, motif_probabilities: np.ndarray) -> Tuple[int, int]:
        motif_matrix = motif_probabilities + np.repeat(motif_probabilities[:, np.newaxis],
                                                       len(motif_probabilities), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            costs = np.divide(motif_matrix, (transition_matrix + transition_matrix.T))
        np.nan_to_num(costs, copy=False, nan=np.inf)  # division by zero could generate nan, so I convert to inf.
        merge_nodes = np.where(costs == costs.min())[-1]
        return merge_nodes[0], merge_nodes[1]

    def build_communities(self,
                          node: str,
                          community_bag: List[List[str]],
                          community_list:
                          Optional[List[str]] = None) -> List[List[str]]:
        if self.tree_ is None:
            raise ValueError("You haven't build the tree.")

        children = list(self.tree_.successors(node))
        if len(children) == 0:
            if community_list is not None:
                community_list.append(node)
            else:
                community_bag.append([node])
        else:
            if self.tree_.nodes.data()[node]["level"] == self.tree_cut:
                traverse_list = []
                for child in children:
                    community_bag = self.build_communities(child, community_bag, traverse_list)
                community_bag.append(traverse_list)
            else:
                for child in children:
                    community_bag = self.build_communities(child, community_bag, community_list)

        return community_bag

    def plot_transition_graph(self,
                              width: str = "500px",
                              height: str = "500px",
                              title: str = "Transition Graph",
                              interactive: bool = False,
                              alpha: float = 0.4) -> Network:
        if self.transition_matrix_ is None:
            raise ValueError("Build transition matrix first.")

        labels = np.arange(self.n_clusters)

        caus_mat = self.transition_matrix_.copy()
        np.fill_diagonal(caus_mat, 0)
        caus_mat[caus_mat < alpha] = 0
        if caus_mat[caus_mat > 0].shape[0] == 0:
            raise Exception("No value is bigger than the threshold alpha: {}".format(alpha))

        G = nx.from_numpy_matrix(caus_mat, parallel_edges=True, create_using=nx.MultiDiGraph())

        label_mapping = {idx: val for idx, val in enumerate(labels)}
        G = nx.relabel_nodes(G, label_mapping)

        nt = Network(height=height, width=width, directed=True, notebook=interactive, heading=title)
        edges = G.edges(data=True)
        nodes = G.nodes(data=True)

        node_size_gain, node_size_offset = 2, 4
        if len(edges) > 0:
            for e in edges:
                nodes[e[0]]['size'] = G.out_degree(e[0]) * node_size_gain + node_size_offset
                nodes[e[1]]['size'] = G.out_degree(e[1]) * node_size_gain + node_size_offset
                nt.add_node(int(e[0]), **nodes[e[0]])
                nt.add_node(int(e[1]), **nodes[e[1]])

                e[2]['value'] = e[2]['weight']
                e[2]['label'] = f"{round(e[2]['value'], 2)}"
                nt.add_edge(int(e[0]), int(e[1]), **e[2])
        return nt

    def plot_tree(self,
                  width: str = "500px",
                  height: str = "500px",
                  title: str = "Binary Tree",
                  interactive: bool = False) -> Network:
        if self.tree_ is None:
            raise ValueError("You haven't build the tree.")

        nt = Network(width=width, height=height, directed=True, layout=True, heading=title, notebook=interactive)
        nt.from_nx(self.tree_)
        return nt
