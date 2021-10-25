import os
import pickle
from typing import Tuple, Optional
from pyDyNet.analytics.dynet import *
from pyDyNet.core.data import MultiNetworks


def create_multi_network(input_networks_directory: str,
                         network_format: str,
                         sep: str = "\t",
                         graph_type: str = "undirected") -> MultiNetworks:
    """ Create multi-network object """
    mn = MultiNetworks()
    if network_format == "adjacency":
        mn.load_adjacency_matrix(input_networks_directory, sep=sep, graph_type=graph_type)
    elif network_format == "edge":
        mn.load_edge_list(input_networks_directory, sep=sep, graph_type=graph_type)
    mn.create_reference_network()
    return mn


def run_dynet(multi_network: MultiNetworks,
              weight_type: str = "node",
              node_list: list = None) -> Tuple[dict, dict, dict]:
    """ Run dynet on the networks """
    adjacency_matrix_list = multi_network.get_adjacency_matrices(weight_type=weight_type)
    reshape_values = get_reshape_values(adj_matrix_list=adjacency_matrix_list)
    weights = extract_weights_across_states(adj_matrix_list=adjacency_matrix_list)
    mean_non_zero_weights_per_state = calculate_non_zero_mean(array=weights, reshape_values=reshape_values)
    std_edge_weights = standardize_edge_weights(weighted_adj_matrix=adjacency_matrix_list,
                                                mean_non_zero_weights_per_state=mean_non_zero_weights_per_state)
    std_edge_weights_by_node = extract_weights_across_states(adj_matrix_list=std_edge_weights)
    centroid_weights = calculate_centroids(std_edge_weights_by_node, reshape_values=reshape_values)
    dn_score = calculate_rewiring_score(std_edge_weights=std_edge_weights,
                                        centroid_weights=centroid_weights,
                                        node_list=node_list)
    n_edges = get_n_edges(network=multi_network.reference_network)
    dn_degree_corrected_scores = calculate_dn_corrected_score(dn_scores=dn_score, n_edges=n_edges)
    return dn_score, n_edges, dn_degree_corrected_scores


def load_node_list(node_list_file_path: str) -> list:
    """ Load node list to a list from file """
    with open(node_list_file_path, "r") as node_list_file:
        node_list = node_list_file.readlines()
    node_list = [node.strip() for node in node_list]
    return node_list


def save_results(output_dir: str,
                 dn_degree_corrected_scores: dict,
                 multi_network: Optional[MultiNetworks],
                 sep: str = '\t') -> None:
    """ Save the results of dynet to the output directory """
    pydynet_results_dir_path = os.path.join(output_dir, "dynet_results")
    if not os.path.exists(pydynet_results_dir_path):
        os.makedirs(pydynet_results_dir_path)
    pydynet_results_file_path = os.path.join(pydynet_results_dir_path, "dn_scores.txt")
    dn_score_df = pd.DataFrame(dn_degree_corrected_scores, index=["dn_score", "n_edges", "dn_score_corrected"]).T
    dn_score_df.to_csv(pydynet_results_file_path, sep=sep, index=False)

    multi_network_save_dir_path = os.path.join(output_dir, "multi_network")
    if not os.path.exists(multi_network_save_dir_path):
        os.makedirs(multi_network_save_dir_path)
    multi_network_save_file_path = os.path.join(multi_network_save_dir_path, "multi_network.pkl")
    with open(multi_network_save_file_path, 'wb') as multi_network_file:
        pickle.dump(multi_network, multi_network_file)


def run_pydynet(input_networks_directory: str,
                network_format: str,
                sep: str,
                graph_type: str,
                weight_type: str,
                output_dir: str,
                node_list: str = None) -> None:
    """ Core execution script for cli call """
    multi_network = create_multi_network(input_networks_directory=input_networks_directory,
                                         network_format=network_format,
                                         sep=sep,
                                         graph_type=graph_type)
    if node_list:
        node_list = load_node_list(node_list_file_path=node_list)
    else:
        node_list = list(multi_network.reference_network.nodes)
    dn_scores, n_edges, dn_degree_corrected_scores = run_dynet(multi_network=multi_network,
                                                               weight_type=weight_type,
                                                               node_list=node_list)
    dn_results = {node: [dn_scores[node], n_edges[node], dn_degree_corrected_scores[node]] for node in dn_scores}
    save_results(output_dir=output_dir,
                 dn_degree_corrected_scores=dn_results,
                 multi_network=multi_network,
                 sep=sep)
