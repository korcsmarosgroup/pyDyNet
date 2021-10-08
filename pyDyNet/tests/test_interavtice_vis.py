import os
import pandas as pd


from ..visualization.interactive_vis import _convert_input_data
from ..visualization.interactive_vis import _draw_interactive_network

for filename in os.listdir("."):
    
    if filename.endswith(".html"):
        os.remove(filename)


def test_convert_input_data():
    node_df = pd.read_csv('pyDyNet/tests/pydynet_example_results/pydynet_example_results/node_dn_scores.tsv', sep='\t', index_col=0)
    edge_df = pd.read_csv('pyDyNet/tests/pydynet_example_results/pydynet_example_results/reference_network_edge_list.tsv', sep='\t')
    results = _convert_input_data(node_df, edge_df)

    assert list(results.nodes) == ['C', 'D', 'B', 'A', 'E']

def test_draw_interactive_network():

    node_df = pd.read_csv('pyDyNet/tests/pydynet_example_results/pydynet_example_results/node_dn_scores.tsv', sep='\t', index_col=0)
    edge_df = pd.read_csv('pyDyNet/tests/pydynet_example_results/pydynet_example_results/reference_network_edge_list.tsv', sep='\t')
    results = _convert_input_data(node_df, edge_df)

    html_results = _draw_interactive_network(results, 800, 800, "output")
    html_counter = 0

    for filename in os.listdir("."):

        if filename.endswith("html"):
            html_counter += 1

    assert html_counter == 1
