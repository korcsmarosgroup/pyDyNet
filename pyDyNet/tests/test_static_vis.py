import os
import matplotlib

import pandas as pd
import networkx as nx

from ..visualization.static_vis import draw_static_network_overview
from ..visualization.static_vis import vis_static

for filename in os.listdir("."):
    
    if filename.endswith(".png"):
        os.remove(filename)


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

    vis_static(G, "test_static.png")
    png_counter = 0

    for filename in os.listdir("."):

        if filename.endswith("png"):
            png_counter += 1

    assert png_counter == 1
