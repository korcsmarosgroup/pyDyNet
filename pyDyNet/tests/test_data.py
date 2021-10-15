import os
import pytest
import numpy as np
import networkx as nx
from distutils import dir_util
from ..core.data import MultiNetworks
from ..core.exception_handler import GraphTypeIncompatible


# Example networks for testing
DIRECTED_NETWORKS = [np.array([[1, 1], [2, 1]]),
                     np.array([[1, 1], [0, 1]]),
                     np.array([[1, 0], [2, 1]])]

UNDIRECTED_NETWORKS = [np.array([[0, 1], [1, 0]]),
                       np.array([[2, 1], [1, 2]]),
                       np.array([[3, 1], [1, 3]])]


@pytest.fixture()
def directed_multi_network():
    """ Create mutli networks from the main data class object """
    mn = MultiNetworks()
    mn.create_networks(network_data=DIRECTED_NETWORKS,
                       data_format="adjacency",
                       graph_type="directed")
    return mn


@pytest.fixture()
def undirected_multi_network():
    """ Create example undirected multi network for testing """
    mn = MultiNetworks()
    mn.create_networks(network_data=UNDIRECTED_NETWORKS,
                       data_format="adjacency",
                       graph_type="undirected")
    return mn


@pytest.fixture
def data_dir(tmpdir, request):
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)
    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))
    return tmpdir


def test_load_edge_list(data_dir):
    """ A test to loading a directory of edge list from files """
    example_edge_list_dir = os.path.join(data_dir, "edge_lists")
    mn = MultiNetworks()
    mn.load_edge_list(example_edge_list_dir, sep='\t', graph_type="undirected")
    actual_matrix = mn.get_adjacency_matrices(weight_type="edge")
    assert len(actual_matrix) == 2
    assert np.array(actual_matrix).shape == (2, 4, 4)


def test_load_adjacency_matrix(data_dir):
    """ A test for loading a directory of adjacency matrices"""
    example_adjacency_matrices_dir = os.path.join(data_dir, "adjacency_matrices")
    mn = MultiNetworks()
    mn.load_adjacency_matrix(example_adjacency_matrices_dir, sep=',', graph_type="directed")
    actual_matrix = mn.get_adjacency_matrices(weight_type="edge")
    expected_case_file_path = os.path.join(example_adjacency_matrices_dir, mn.state_names[0])
    expected_matrix = np.loadtxt(expected_case_file_path, delimiter=',')
    assert len(actual_matrix) == 2
    assert np.array(actual_matrix).shape == (2, 3, 3)
    assert np.allclose(expected_matrix, actual_matrix[0])


def test_incompatible_graph_type():
    """ Test that a directed network cannot be entered as a undirected network """
    with pytest.raises(GraphTypeIncompatible):
        mn = MultiNetworks()
        mn.create_networks(network_data=DIRECTED_NETWORKS,
                           data_format='adjacency',
                           graph_type="undirected")


@pytest.mark.parametrize("weight_type,graph_type", [('edge', 'undirected'),
                                                    ('node', 'undirected'),
                                                    ('edge', 'directed'),
                                                    ('node', 'directed')])
def test_get_adjacency_matrix(directed_multi_network, undirected_multi_network, weight_type, graph_type):
    """ Test to extract the adjacency matrices for both node and edge weights """
    if graph_type == "undirected":
        mn = directed_multi_network
        adjacency_matrix = mn.get_adjacency_matrices(weight_type=weight_type)
        expected_network = DIRECTED_NETWORKS
    else:
        mn = undirected_multi_network
        adjacency_matrix = mn.get_adjacency_matrices(weight_type=weight_type)
        expected_network = UNDIRECTED_NETWORKS
    if weight_type == "edge":
        assert (np.array(adjacency_matrix) == np.array(expected_network)).all()
    else:
        assert not ((np.array([mat > 1 for mat in adjacency_matrix])).reshape(1, -1).any())
    assert len(DIRECTED_NETWORKS) == len(adjacency_matrix)
    assert np.array(adjacency_matrix).shape == np.array(expected_network).shape


def test_create_networks_adjacency(directed_multi_network):
    """ Generate networks from input list of adjacency matrices """
    mn = directed_multi_network
    assert len(mn.network_list) == 3
    assert DIRECTED_NETWORKS[0].shape == (2, 2)


def test_create_reference_network_directed(directed_multi_network):
    """ Test of the creation of the reference network """
    mn = directed_multi_network
    mn.create_reference_network()
    assert len(list(mn.reference_network.nodes)) == 2
    assert len(list(mn.reference_network.edges)) == 3
    assert isinstance(mn.reference_network, nx.classes.graph.Graph)


def test_create_reference_network_undirected(undirected_multi_network):
    """ Test for the creation of the reference network """
    mn = undirected_multi_network
    mn.create_reference_network()
    assert len(mn.reference_network.nodes) == 2
    assert list(mn.reference_network.edges) == [(0, 1), (0, 0), (1, 1)]

