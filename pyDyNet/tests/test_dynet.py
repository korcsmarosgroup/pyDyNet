import os
import numpy as np
import pytest
from distutils import dir_util
from pyDyNet.analytics.dynet import extract_weights_across_states, calculate_non_zero_mean, get_reshape_values, \
    standardize_edge_weights, calculate_centroids, calculate_rewiring_score, get_n_edges, calculate_dn_corrected_score
from pyDyNet.core.data import MultiNetworks


# Expected rewiring score results
EXPECTED_DN_SCORES = {0: 1.6887276850983164,
                      1: 2.0628638745095786,
                      2: 1.6965244372012631,
                      3: 1.8574723292996402,
                      4: 1.7800983031070388}

EXPECTED_DN_CORRECTED_SCORES = {0: 0.33774553701966326,
                                1: 0.41257277490191574,
                                2: 0.3393048874402526,
                                3: 0.37149446585992807,
                                4: 0.3560196606214078}


@pytest.fixture
def data_dir(tmpdir, request):
    """ A helper function for copying example and test data to the tmpdir """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)
    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))
    return tmpdir


@pytest.fixture()
def create_network(data_dir):
    """ Create example undirected multi network for testing """
    example_input_data = os.path.join(data_dir, "raw_data")
    mn = MultiNetworks()
    mn.load_adjacency_matrix(example_input_data, sep='\t', graph_type="undirected")
    return mn


def test_extract_weights_across_states(create_network):
    """ Testing extracting the weights and creating a 2D array """
    mn = create_network
    adj_matrix_list = mn.get_adjacency_matrices(weight_type="edge")
    weights = extract_weights_across_states(adj_matrix_list=adj_matrix_list)
    assert weights.shape == (25, 5)
    assert len(weights.shape) == 2


def test_calculate_non_zero_mean(data_dir, create_network):
    """ Testing calculation of non-zero mean of the weights over all states"""
    actual_non_zero_mean_file = os.path.join(data_dir, "non_zero_mean", "mean_non_zero_edges.txt")
    mn = create_network
    adj_matrix_list = mn.get_adjacency_matrices(weight_type="edge")
    weights = extract_weights_across_states(adj_matrix_list=adj_matrix_list)
    reshape_values = get_reshape_values(adj_matrix_list=adj_matrix_list)
    mean_non_zero_weights_per_state = calculate_non_zero_mean(array=weights, reshape_values=reshape_values)
    actual_non_zero_mean = np.loadtxt(fname=actual_non_zero_mean_file, delimiter="\t")
    assert reshape_values == (5, 5)
    assert np.allclose(np.around(mean_non_zero_weights_per_state, 2), actual_non_zero_mean)


def test_standardize_edge_weights(data_dir, create_network):
    """ Testing calculating the standardized edge weights """
    mn = create_network
    adj_matrix_list = mn.get_adjacency_matrices(weight_type="edge")
    actual_non_zero_mean_file = os.path.join(data_dir, "non_zero_mean", "mean_non_zero_edges.txt")
    expected_std_edge_weights_folder = os.path.join(data_dir, "standardized_data")
    expected_std_edge_weights_files = [os.path.join(expected_std_edge_weights_folder, f)
                                       for f in os.listdir(expected_std_edge_weights_folder)]
    mean_non_zero_weights_per_state = np.loadtxt(fname=actual_non_zero_mean_file, delimiter="\t")
    expected_std_edge_weights = np.array([np.loadtxt(fname=f, delimiter="\t") for f in expected_std_edge_weights_files])
    actual_std_edge_weights = standardize_edge_weights(weighted_adj_matrix=adj_matrix_list,
                                                       mean_non_zero_weights_per_state=mean_non_zero_weights_per_state)
    assert np.allclose(np.around(actual_std_edge_weights, 2)[0],
                       np.around(expected_std_edge_weights, 2)[0])


def test_calculate_centroids(data_dir, create_network):
    """ Testing calculating the centroids for the edge weights """
    mn = create_network
    adj_matrix_list = mn.get_adjacency_matrices(weight_type="edge")
    reshape_values = get_reshape_values(adj_matrix_list=adj_matrix_list)
    weights = extract_weights_across_states(adj_matrix_list=adj_matrix_list)
    mean_non_zero_weights_per_state = calculate_non_zero_mean(array=weights, reshape_values=reshape_values)
    actual_std_edge_weights = standardize_edge_weights(weighted_adj_matrix=adj_matrix_list,
                                                       mean_non_zero_weights_per_state=mean_non_zero_weights_per_state)
    std_edge_weights_by_node = extract_weights_across_states(adj_matrix_list=actual_std_edge_weights)
    actual_centroid_weights = calculate_centroids(std_edge_weights_by_node, reshape_values=reshape_values)
    expected_centroid_weights_file = os.path.join(data_dir, "centroids", "centroids.txt")
    expected_centroid_weights = np.loadtxt(fname=expected_centroid_weights_file, delimiter="\t")
    assert np.allclose(np.around(actual_centroid_weights, 2), expected_centroid_weights)
    assert actual_centroid_weights.shape == expected_centroid_weights.shape


def test_calculate_rewiring_score(create_network):
    """ Testing the calculation of the rewiring score """
    mn = create_network
    adj_matrix_list = mn.get_adjacency_matrices(weight_type="edge")
    reshape_values = get_reshape_values(adj_matrix_list=adj_matrix_list)
    weights = extract_weights_across_states(adj_matrix_list=adj_matrix_list)
    mean_non_zero_weights_per_state = calculate_non_zero_mean(array=weights, reshape_values=reshape_values)
    std_edge_weights = standardize_edge_weights(weighted_adj_matrix=adj_matrix_list,
                                                mean_non_zero_weights_per_state=mean_non_zero_weights_per_state)
    std_edge_weights_by_node = extract_weights_across_states(adj_matrix_list=std_edge_weights)
    centroid_weights = calculate_centroids(std_edge_weights_by_node, reshape_values=reshape_values)
    actual_dn_score = calculate_rewiring_score(std_edge_weights=std_edge_weights,
                                               centroid_weights=centroid_weights,
                                               node_list=[0, 1, 2, 3, 4])
    assert actual_dn_score == EXPECTED_DN_SCORES


def test_calculate_dn_corrected_score(create_network):
    """ Testing the degree correction function on the dn scores for rewiring """
    mn = create_network
    mn.create_reference_network()
    actual_n_edges = get_n_edges(network=mn.reference_network)
    actual_dn_degree_corrected_scores = calculate_dn_corrected_score(dn_scores=EXPECTED_DN_SCORES,
                                                                     n_edges=actual_n_edges)
    actual_dn_degree_corrected_scores = np.array(list(actual_dn_degree_corrected_scores.items()))
    expected_dn_corrected_scores = np.array(list(EXPECTED_DN_CORRECTED_SCORES.items()))
    assert np.allclose(np.around(actual_dn_degree_corrected_scores, 2),
                       np.around(expected_dn_corrected_scores, 2))
