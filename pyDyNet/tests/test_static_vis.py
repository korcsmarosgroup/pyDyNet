import os


import pandas as pd
import networkx as nx
from ..visualization.static_vis import draw_static_network_overview

for filename in os.listdir("."):
    
    if filename.endswith(".png"):
        os.remove(filename)


def test_draw_static_network_overview():

    edges = pd.read_csv('pyDyNet/tests/vis_tests_files/edgetable.txt')
    nodes = pd.read_csv('pyDyNet/tests/vis_tests_files/nodetable.txt')
    G = nx.from_pandas_edgelist(edges, 'source', 'target')
    G.add_nodes_from(nodes.set_index('name').to_dict('index').items())

    results = draw_static_network_overview(G, node_sizes = "Dn-Score (degree corrected)", main_title="An example plot of corrected scores")
    png_counter = 0

    for filename in os.listdir("."):

        if filename.endswith("png"):
            png_counter += 1

    assert png_counter == 1


def test_vis_static():

    edges = pd.read_csv('pyDyNet/tests/pydynet_example_results/pydynet_example_results/reference_network_edge_list.tsv', sep="\t")
    nodes = pd.read_csv('pyDyNet/tests/pydynet_example_results/pydynet_example_results/node_dn_scores.tsv', sep="\t")
    G = nx.from_pandas_edgelist(edges, 'source', 'target')
    G.add_nodes_from(nodes.set_index('Unnamed: 0').to_dict('index').items())

    results = vis_static(G)
    png_counter = 0

    for filename in os.listdir("."):

        if filename.endswith("png"):
            png_counter += 1

    assert png_counter == 1
