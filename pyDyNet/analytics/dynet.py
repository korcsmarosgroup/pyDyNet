import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict


def create_union(network_list: list) -> nx.classes.graph.Graph:
    """ Create union of graphs """
    union_edge_list = []
    for network in network_list:
        _edge_list = list(nx.to_edgelist(network))
        _edge_list = [i[0:2] for i in _edge_list]
        union_edge_list.extend(_edge_list)
    union_network = list(set(union_edge_list))
    g = nx.Graph()
    g.add_edges_from(union_network)
    return g


def reorder_nodes(network: nx.classes.Graph,
                  nodes_sorted: list) -> pd.DataFrame:
    """ Return a adjancy list with a sorted node structure """
    adj_list = nx.to_pandas_adjacency(network)
    sorted_adj_list = adj_list.reindex(index=nodes_sorted, columns=nodes_sorted)
    return sorted_adj_list


def iter_pandas_to_array(matrix_df_list: list) -> list:
    """ Extract np array from pandas frames """
    return [frame.values for frame in matrix_df_list]


def extract_weights_across_states(adj_matrix_list: np.array) -> np.array:
    """ Extract each node/edge weight pair across adjancey matrices """
    node_edge_weight = []
    for k in range(adj_matrix_list.shape[0]):
        for j in range(adj_matrix_list.shape[1]):
            current_node_weights = []
            for i in range(adj_matrix_list.shape[2]):
                _edge_weight = adj_matrix_list[i][j][k]
                current_node_weights.append(np.array(_edge_weight))
            node_edge_weight.append(current_node_weights)
    node_edge_weight = np.array(node_edge_weight)
    return node_edge_weight


def calculate_non_zero_mean(array: np.array,
                            reshape_values: tuple,
                            axis: int = 1) -> np.array:
    """ Extract the non-zero mean across 2D array """
    mean_non_zero_weights = np.true_divide(array.sum(axis), (array != 0).sum(axis))
    mean_non_zero_weights = mean_non_zero_weights.reshape(reshape_values)
    return mean_non_zero_weights


def calculate_centroids(array: np.array,
                        reshape_values: tuple,
                        axis: int = 1) -> np.array:
    """ Extract the non-zero mean across 2D array """
    centroid_weights = np.mean(array, axis=axis, dtype=np.float64)
    centroid_weights = centroid_weights.reshape(reshape_values)
    return centroid_weights


def calculate_non_zero_centroids(array: np.array,
                                 reshape_values: tuple,
                                 axis: int = 1) -> np.array:
    """ Extract the non-zero mean across 2D array """
    non_zero_centroid_weights = np.true_divide(array, axis=axis, dtype=np.float64)
    non_zero_centroid_weights = non_zero_centroid_weights.reshape(reshape_values)
    return non_zero_centroid_weights


def standardize_edge_weights(weighted_adj_matrix: np.array,
                             mean_non_zero_weights_per_state: np.array) -> np.array:
    """ Standardize the edge weights by the non zero mean weights """
    standardise_edge_weights_matrices = []
    for layer in range(np.array(weighted_adj_matrix).shape[0]):
        standardise_edge_weights_matrices.append(weighted_adj_matrix[layer] /
                                                 mean_non_zero_weights_per_state)
    standardise_edge_weights_matrices = np.array(standardise_edge_weights_matrices)
    return standardise_edge_weights_matrices


def get_degree(network: nx.classes.graph.Graph) -> dict:
    """ Extract the degree for each node from the network """
    return {node: val for (node, val) in network.degree()}


def get_n_edges(network: nx.classes.graph.Graph) -> dict:
    """ Get the number of edges for all nodes in a network """
    return {node: len(network.edges(node)) for node in list(network.nodes())}


def calculate_rewiring_score(std_edge_weights: np.array,
                             centroid_weights: np.array,
                             node_list: list) -> dict:
    """ Calculate the euclidean distance from node to centroid """
    results = defaultdict(list)
    for layer in range(std_edge_weights.shape[0]):
        for row in range(std_edge_weights.shape[1]):
            dist = (std_edge_weights[layer][row] - centroid_weights[row]) * (
                    std_edge_weights[layer][row] - centroid_weights[row])
            result = np.sqrt(sum(dist))
            node = node_list[row]
            results[node].append(result)
    dn_scores = {node: (np.sum(results[node]) / (std_edge_weights.shape[1] - 1)) for node in results}
    return dn_scores


def calculate_dn_corrected_score(dn_scores: dict,
                                 n_edges: dict) -> dict:
    """ Calculate the corrected dn score values """
    assert dn_scores.keys() == dn_scores.keys()
    return {node: dn_scores[node] / (n_edges[node]) for node in dn_scores}
