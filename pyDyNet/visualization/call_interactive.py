from pyDyNet.visualization import interactive_vis
import pandas as pd


def vis_interactive(node_df, edge_df, out_filename='output/interactive_graph_out', xsize=800, ysize=800):
    """
    Call interactive vis functions
    :param node_df: dataframe of nodes and dn scores
    :param edge_df: dataframe of edges
    :param xsize: size of output html graph
    :param ysize: size of output html graph
    :param out_filename: output file name
    :return: .html interactive graph
    """

    G = interactive_vis.convert_input_data(node_df, edge_df)
    int_vis_out = interactive_vis.draw_interactive_network(G, outfile=out_filename, xsize=xsize, ysize=ysize)

    return int_vis_out

