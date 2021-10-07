import os


import pandas as pd
import networkx as nx
from ..visualization.interactive_vis import convert_input_data
from ..visualization.static_vis import draw_static_network_overview

for filename in os.listdir("."):
    
    if filename.endswith(".png"):
        os.remove(filename)


def test_draw_static_network_overview():

    edges = pd.read_csv('edgetable.txt')
    nodes = pd.read_csv('nodetable.txt')
    G = nx.from_pandas_edgelist(edges, 'source', 'target')
    G.add_nodes_from(nodes.set_index('name').to_dict('index').items())

    results = draw_static_network_overview(G, "Dn-Score (degree corrected)", main_title="An example plot of corrected scores")
    png_counter = 0

    for filename in os.listdir("."):

        if filename.endswith("png"):
            png_counter += 1

    assert png_counter == 1
