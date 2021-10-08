import matplotlib
from matplotlib import pyplot as plt

from typing import Union, Callable

import numpy as np
import pandas as pd
import networkx as nx

def draw_netstat_histo(
        values: list,
        label: str,
        ax: Union[None, plt.axes] = None,
) -> tuple:
    """
    Draw a histogram of one of the visual channels (color or size of nodes).

    Parameters
    ----------
    values
        Values whose distribution will be shown
    label
        Information on what values are shown
    ax
        A matplotlib axis where the plot needs to be drawn

    Returns
    -------
    The matplotlib axis with histogram.
    """

    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(values, color="silver")
    ax.set_xlabel(label)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax


def draw_netstat_scatter(
        x_values: list,
        y_values: list,
        labels: tuple,
        ax: Union[None, plt.axes] = None,
) -> tuple:
    """
    Draw a histogram of one of the visual channels (color or size of nodes).

    Parameters
    ----------
    x_values
        Values for the x axis
    y_values
        Values for the y axis
    labels
        Information on what values are shown: x label and y label
    ax
        A matplotlib axis where the plot needs to be drawn

    Returns
    -------
    The matplotlib axis with histogram.
    """

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(x_values, y_values, color="red", s=3, alpha=0.5)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax


def nx_node_features_to_pd(
        network: nx.classes.graph.Graph
) -> pd.DataFrame:
    """
    Helper to convert node features into a dataframe for easier access.

    Parameters
    ----------
    network
        The network with node attributes

    Returns
    -------
    A table of node features.
    """

    node_features = pd.DataFrame.from_dict(network.nodes, orient="index").reset_index()

    return node_features


def draw_static_network(
        network: nx.classes.graph.Graph,
        node_features: Union[None, pd.DataFrame] = None,
        scores: str = "dn_score",
        node_sizes: str = "n_edges",
        node_list: Union[None, list] = None,
        node_labels: Union[None, str, dict] = None,
        node_positions: Union[None, dict] = None,
        node_size_factor: Union[int, float, Callable] = 50,
        node_colormap: matplotlib.colors.LinearSegmentedColormap = plt.cm.Reds,
        color_min_max: Union[None, tuple] = None,
        colorbar: bool = True,
        graph_title: str = "",
        ax: Union[None, plt.axes] = None,
        **networkx_kwds
) -> plt.axes:
    """
    Draw a static network with nodes colored by a given score.

    Parameters
    ----------
    network
        A networkx object with scores added ad node features.
    node_features
        If node features are supplied, we can save time converting the network to pandas.
    scores
        The scores that will determine node colors. Has to be a column name in node features.
    node_list
        Set of nodes to show; its order matters here
    node_sizes
        Name of node attribute (e.g. number of edges) that will correspond to node size.
    node_labels
        A mapping of node labels if it is not identical to the node index.
    node_positions
        A mapping of node positions; fixing positions between samples can help comparison.
    node_size_factor
        A multiplier that is applied to the node sizes over the score values.
    node_colormap
        A colormap that will be used to show scores.
    color_min_max
        By default, the color scale spans from minimum score to maximum. Provide a tuple of
        min and max values to override this.
    colorbar
        If a colorbar should be added.
    graph_title
        A title put on the plot.
    ax
        The pyplot axis if the network should be drawn on a predefined axis.
    networkx_kwds
        A dictionary of keywords passed over to `draw_networkx`.

    Returns
    -------
    A matplotlib axis where the network was drawn.
    """

    # Keep a possibility to supply a node_list by which the big network will be filtered
    if node_list is not None:
        nodes_in_network = list(network)
        node_list = pd.Series(node_list)
        node_list = node_list.loc[node_list.isin(nodes_in_network)]
        network = nx.subgraph(network, node_list)
        if node_features is None:
            node_features = nx_node_features_to_pd(network)
        else:
            nodes_in_network = list(network)
            node_features = node_features.loc[nodes_in_network]
    else:
        if node_features is None:
            node_features = nx_node_features_to_pd(network)

    # For better comparison, fixed node positions might be supplied
    if node_positions is None:
        node_positions = nx.spring_layout(network, k=0.08)

    # Usual case would be to supply a linear scaling factor for node sizes, but it is also possible to scale non-linearly
    if isinstance(node_size_factor, (int, float)):
        node_sizes = np.array(node_features[node_sizes]) * node_size_factor
    else:
        node_sizes = node_size_factor(node_features[node_sizes])
    
    scores = node_features[scores]

    # Check if user supplied colorscale limits
    if color_min_max is None:
        vmin = min(scores)
        vmax = max(scores)
    else:
        if color_min_max[0] is None:
            vmin = min(scores)
        else:
            vmin = color_min_max[0]
        if color_min_max[1] is None:
            vmax = max(scores)
        else:
            vmax = color_min_max[1]

    # In most cases, the function should draw on an already existing axis, but let it handle missing case
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_title(graph_title)

    # Finally, draw the network graph
    nx.draw_networkx(
        network, pos=node_positions, node_color=scores, node_size=node_sizes,
        labels=node_labels, edge_color="silver", cmap=node_colormap,
        vmin=vmin, vmax=vmax, ax=ax, **networkx_kwds
    )

    # Add a colorbar to the axis if needed
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=node_colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm)
    
    # Remove spines
    ax.axis('off')

    return ax


def draw_static_network_overview(
        network: nx.classes.graph.Graph,
        node_features: Union[None, pd.DataFrame] = None,
        scores: str = "dn_score",
        node_sizes: str = "n_edges",
        node_labels: Union[None, str, dict] = None,
        node_positions: Union[None, dict] = None,
        node_size_factor: Union[int, float, Callable] = 50,
        node_colormap: matplotlib.colors.LinearSegmentedColormap = plt.cm.Reds,
        color_min_max: Union[None, tuple] = None,
        main_title: str = "",
        show_stats: bool = True,
        figsize: tuple = (7.2, 9.6)
) -> tuple:
    """
    Draw an overview of the scored network. Static network with nodes
    colored by score, plus the distribution of scores and their correlation
    with degree.

    Parameters
    ----------
    network
        A networkx object with scores added ad node features.
    node_features
        If node features are supplied, we can save time converting the network to pandas.
    scores
        The scores that will determine node colors. Has to be a column name in node features.
    node_sizes
        Name of node attribute (e.g. number of edges) that will correspond to node size.
    node_labels
        A mapping of node labels if it is not identical to the node index.
    node_positions
        A mapping of node positions; fixing positions between samples can help comparison.
    node_size_factor
        A multiplier that is applied to the node sizes over the score values.
    node_colormap
        A colormap that will be used to show scores.
    color_min_max
        By default, the color scale spans from minimum score to maximum. Provide a tuple of
        min and max values to override this.
    main_title
        Title to be put above the whole panel.
    show_stats
        If the two extra plots should also be shown or just the network.
    figsize
        Figure size; just to aim big, it defaults to a full-page A4 figure.

    Returns
    -------
    A tuple of matplotlib axes.
    """

    node_list = None
    if node_features is None:
        node_features = nx_node_features_to_pd(network)

    # If some descriptive stats for scores and degrees are to be shown, it makes sense to access them now
    if show_stats:

        fig, axs = plt.subplots(
            2, 3, figsize=figsize,
            gridspec_kw={"width_ratios": (1, 1, 1), "height_ratios": (4, 1)}
        )
        gs = axs[0, 0].get_gridspec()
        for ax in axs[0, 0:]:
            ax.remove()
        ax_net = fig.add_subplot(gs[0, 0:])

        draw_static_network(
            network, node_features, scores, node_sizes, node_labels, node_positions,
            node_list, node_size_factor, node_colormap, color_min_max, ax=ax_net
        )

        scores_l = node_features[scores]
        node_sizes_l = node_features[node_sizes]

        draw_netstat_histo(scores_l, label="Color", ax=axs[1, 0])
        draw_netstat_scatter(scores_l, node_sizes_l, labels=("Color", "Size"), ax=axs[1, 1])
        draw_netstat_histo(node_sizes_l, label="Size", ax=axs[1, 2])

    # If stats plots are not required, we can leave everything to the static network drawer
    else:
        fig, axs = plt.subplots(figsize=figsize)
        draw_static_network(
            network, node_features, scores, node_sizes, node_labels, node_positions,
            node_list, node_size_factor, node_colormap, color_min_max, ax=axs
        )

    fig.suptitle(main_title)

    return axs


def vis_static(
        network: nx.classes.graph.Graph,
        out_filename: str = "output/static_graph_overview.png",
        scores: str = "dn_score",
        node_sizes: str = "n_edges",
        main_title: str = "",
        show_stats: bool = True,
        figsize: tuple = (7.2, 9.6)
) -> None:
    """
    High-level function to draw an overview of the scored network and save the output.

    Parameters
    ----------
    network
        A networkx object with scores added ad node features.
    out_filename
        Path where figure needs to be saved.
    scores
        The scores that will determine node colors. Has to be a column name in node features.
    node_sizes
        Name of node attribute (e.g. number of edges) that will correspond to node size.
    main_title
        Title to be put above the whole panel.
    show_stats
        If the two extra plots should also be shown or just the network.
    figsize
        Figure size; just to aim big, it defaults to a full-page A4 figure.

    Returns
    -------
    Does not return anything (an IO stream later?)
    """

    axs = draw_static_network_overview(
        network=network, scores=scores, node_sizes=node_sizes, main_title=main_title,
        show_stats=show_stats, figsize=figsize
     )
    
    fig = axs.flatten()[-1].get_figure()
    fig.savefig(out_filename)

    return