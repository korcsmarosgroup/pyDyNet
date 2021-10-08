import networkx as nx
from pyvis.network import Network
import pandas as pd


def convert_input_data(nodes, edges):
    """
    Assign colors to each node based on Dn-score
    :param nodelist: file containing nodes and Dn scores
    :param edgelist: file containing edges
    :return: networkx graph object
    """
    # rescaling the rewiring values to RGB
    a, b = 255, 0
    x = nodes['dn_score'].min()
    y = nodes['dn_score'].max()

    # Save as separate columns
    nodes['R'] = 255
    nodes['G'] = (nodes['dn_score'] - x) / (y - x) * (b - a) + a
    nodes['G'] = nodes['G'].round().astype(int)
    nodes['B'] = (nodes['dn_score'] - x) / (y - x) * (b - a) + a
    nodes['B'] = nodes['B'].round().astype(int)

    nodes = nodes[['dn_score', 'dn_score_corrected', 'n_edges', 'R', 'G', 'B']]
    # Maybe a python magician could rewrite this list comp to be more efficient
    nodes['color'] = ['#%02x%02x%02x' % (nodes['R'][i], nodes['G'][i], nodes['B'][i]) for i in range(len(nodes))]

    # rescaling the rewiring values to size
    c, d = 15, 50
    nodes['value'] = (nodes['dn_score'] - x) / (y - x) * (d - c) + c
    nodes['value'] = nodes['value'].round().astype(int)

    # Title controls hover behaviour in pyvis
    nodes['title'] = ('name: ' + nodes.index.astype('str')) + ' Dn: ' + (
        nodes['dn_score'].astype('str'))
    # Create NetworkX object

    G = nx.from_pandas_edgelist(edges, 'source', 'target')
    data = nodes.to_dict('index').items()
    G.add_nodes_from(data)

    return (G)

def draw_interactive_network(
            network,
            xsize,
            ysize,
            outfile,
            jupyter=False):
    """
    Creates interactive network from networkx object
    :param network: networkx object, output of convert_input_data
    :param xsize: pixel size of x axis of interactive figure
    :param ysize: pixel size of y axis of interactive figure
    :param jupyter: True/False value wether to display in jupyter notebook - default set to False
    :param outfile: location and name of output .html file
    :return: .html interative plot
    """
    # set up canvas
    nt = Network(f'{xsize}px', f'{ysize}px', notebook=True)
    # load network data
    nt.from_nx(network)
    # layout
    nt.force_atlas_2based()
    nt.inherit_edge_colors(status=False)
    # draw
    return nt.show(f'{outfile}.html')
