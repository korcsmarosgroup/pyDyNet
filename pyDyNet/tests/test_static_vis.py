import os
import pickle

import pandas as pd
import networkx as nx

from ..visualization.static_vis import draw_subnetwork_facets
from ..visualization.static_vis import draw_static_network_overview
from ..visualization.static_vis import vis_static

for filename in os.listdir("."):
    
    if filename.endswith(".png"):
        os.remove(filename)


def draw_subnetwork_facets():

    edges = pd.read_csv('pyDyNet/tests/pydynet_example_results/pydynet_example_results/reference_network_edge_list.tsv', sep="\t")
    nodes = pd.read_csv('pyDyNet/tests/pydynet_example_results/pydynet_example_results/node_dn_scores.tsv', sep="\t")
    G = nx.from_pandas_edgelist(edges, 'source', 'target')
    G.add_nodes_from(nodes.set_index('Unnamed: 0').to_dict('index').items())

    top_nodes = nodes.sort_values(by="dn_score").tail(2)
    top_nodes = top_nodes.index.tolist()

    with open('../pydynet_example_results/pydynet_nx_network_list.pkl','rb') as infile:
        input_graph_collection = pickle.load(infile)
        input_graph_names = ["Graph_" + str(x) for x in range(1, len(input_graph_collection)+1)]

    results = draw_subnetwork_facets(G, input_graph_collection, input_graph_names, nodelist=top_nodes)
    result = results[0, 0]
    
    assert result.__class__.__name__ == 'AxesSubplot'


def test_draw_static_network_overview():

    edges = pd.read_csv('pyDyNet/tests/pydynet_example_results/pydynet_example_results/reference_network_edge_list.tsv', sep="\t")
    nodes = pd.read_csv('pyDyNet/tests/pydynet_example_results/pydynet_example_results/node_dn_scores.tsv', sep="\t")
    G = nx.from_pandas_edgelist(edges, 'source', 'target')
    G.add_nodes_from(nodes.set_index('Unnamed: 0').to_dict('index').items())

    results = draw_static_network_overview(G, main_title="An example plot of corrected scores")
    result = results[0, 0]
    
    assert result.__class__.__name__ == 'AxesSubplot'


def test_vis_static():

    edges = pd.read_csv('pyDyNet/tests/pydynet_example_results/pydynet_example_results/reference_network_edge_list.tsv', sep="\t")
    nodes = pd.read_csv('pyDyNet/tests/pydynet_example_results/pydynet_example_results/node_dn_scores.tsv', sep="\t")
    G = nx.from_pandas_edgelist(edges, 'source', 'target')
    G.add_nodes_from(nodes.set_index('Unnamed: 0').to_dict('index').items())

    vis_static(G, "vis_static_test.png")
    png_counter = 0

    for filename in os.listdir("."):

        if filename.endswith("png"):
            png_counter += 1

    assert png_counter == 1
