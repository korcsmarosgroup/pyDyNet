### Tools to be used
import matplotlib
from typing import Union, Callable
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx


### Start from an edge table

#edges = pd.read_csv('../rewiring_test_data/edgetable.csv')
#edges.head()

### Rewiring scores are node features

#nodes = pd.read_csv('../rewiring_test_data/nodetable.csv')
#nodes.head()

### Create network object as major expected input

#G = nx.from_pandas_edgelist(edges, 'source', 'target')
#G.add_nodes_from(nodes.set_index('name').to_dict('index').items())


def draw_netstat_histo(
        values: list,
        label: str,
        ax: Union[None, plt.axes] = None,
) -> tuple:
    """
    Draw a histogram of one of the visual chanels (color or size of nodes).

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
    Draw a histogram of one of the visual chanels (color or size of nodes).

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


def check_node_feature_integrity(
        network: nx.classes.graph.Graph,
        node_list: list
) -> tuple:
    """
    Helper to check integrity between user supplied node list and the nodes
    in the network.

    Parameters
    ----------
    network
        The network that will be shown
    node_list
        Set of nodes to show; its order matters here

    Returns
    -------
    The (filtered) network and the reconciled list of labels.
    """

    nodes_in_network = list(network)

    # As we are accepting arrays of values from outside the networkx object,
    # implement some sort of filtering to avoid setting non-existent nodes
    if node_list is None:
        node_list = nodes_in_network
    else:
        node_list = pd.Series(node_list)
        node_list = node_list.loc[node_list.isin(nodes_in_network)]
        network = nx.subgraph(network, node_list)

    return network, node_list


def get_node_labels(
        node_labels: Union[None, str, list, dict],
        node_features: pd.DataFrame,
        node_list: list
) -> Union[None, dict]:
    """
    Helper to standardize user input of node labels.

    Parameters
    ----------
    node_labels
        User supplied node labels
    node_features
        Node features in tabular format for easier access (than the networkx format)
    node_list
        Set of nodes to show; its order matters here

    Returns
    -------
    A mapping between node IDs and the labels to show.
    """

    # Node IDs might not be what we want to show, so check if user supplied custom values
    if node_labels is not None:
        if isinstance(node_labels, str):
            node_labels = node_features.set_index("index").loc[:, node_labels].to_dict()
        else:
            if not isinstance(node_labels, dict):
                node_labels = dict(zip(node_list, node_labels))
    return node_labels


def get_node_scores(
        scores: Union[str, list],
        node_features: pd.DataFrame,
        node_list: list
) -> Union[None, dict]:
    """
    Helper to standardize user input of node scores.

    Parameters
    ----------
    scores
        User supplied node weights
    node_features
        Node features in tabular format for easier access (than the networkx format)
    node_list
        Set of nodes to show; its order matters here

    Returns
    -------
    A list of numerical values that will impact on node colors.
    """

    if isinstance(scores, str):
        scores = node_features[scores]
    else:
        if isinstance(scores, dict):
            scores = [scores[x] for x in node_list]

    return scores


def get_node_sizes(
        node_sizes: Union[None, str, list, dict],
        node_features: pd.DataFrame,
        node_list: list
) -> Union[None, dict]:
    """
    Helper to standardize user input of node size weights.

    Parameters
    ----------
    node_sizes
        User supplied value if node size correspond to something else than degree
    node_features
        Node features in tabular format for easier access (than the networkx format)
    node_list
        Set of nodes to show; its order matters here

    Returns
    -------
    A list of numerical values that will impact on node colors.
    """

    if isinstance(node_sizes, str):
        node_sizes = node_features[node_sizes]
    else:
        if isinstance(node_sizes, dict):
            node_sizes = [node_sizes[x] for x in node_list]

    return node_sizes


def draw_static_network(
        network: nx.classes.graph.Graph,
        scores: Union[str, list],
        node_list: Union[None, list] = None,
        node_sizes: Union[None, str, list, dict] = None,
        node_labels: Union[None, str, list, dict] = None,
        node_positions: Union[None, dict] = None,
        node_size_factor: Union[int, float, Callable] = 50,
        node_colormap: matplotlib.colors.LinearSegmentedColormap = plt.cm.Reds,
        color_min_max: Union[None, tuple] = None,
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
    scores
        The scores that will determine node colors. If a single string is given, then it is
        assumed to be the name of the attribute; if a list than those numbers in the list
        (not recommended as it must be same order as nodelist or mapping is not guarantied)
    node_list
        List of nodes to show; useful if we want to prefilter based on score or pathway
    node_sizes
        A list, or even better, dictionary of node sizes; using degree if not specified
    node_labels
        A mapping of node labels if it is not identical to the node index
    node_positions
        A mapping of node positions; fixing positions between samples can help comparison
    node_size_factor
        A multiplier that is applied to the node sizes over the score values
    node_colormap
        A colormap that will be used to show scores
    color_min_max
        By default, the color scale spans from minimum score to maximum. Provide a tuple of
        min and max values to override this
    graph_title
        A title put on the plot
    ax
        The pyplot axis if the network should be drawn on a predefined axis
    networkx_kwds
        A dictionary of keywords passed over to `draw_networkx`

    Returns
    -------
    A matplotlib axis where the network was drawn.
    """

    # As visualisation heavily relies on node features and node order, try to reconsiliate node list
    network, node_list = check_node_feature_integrity(network, node_list)

    # We will use node features a lot, so easiest is to retreive them as a dataframe
    node_features = pd.DataFrame.from_dict(network.nodes, orient="index").reset_index()

    # Node IDs might not be what we want to show, so check if user supplied custom values
    node_labels = get_node_labels(node_labels, node_features, node_list)

    # Scores can be supplied by the user in various ways, so convert it to a list of right order
    scores = get_node_scores(scores, node_features, node_list)

    # Just like scores, sizes might also be specified upstream, so check
    if node_sizes is None:
        node_sizes = dict(network.degree)
    node_sizes = get_node_sizes(node_sizes, node_features, node_list)

    # Usual case would be to supply a linear scaling factor for node sizes, but it is also possible to scale non-linearly
    if isinstance(node_size_factor, (int, float)):
        node_sizes = np.array(node_sizes) * node_size_factor
    else:
        node_sizes = node_size_factor(node_sizes)

    # For better comparison, fixed node positions might be supplied
    if node_positions is None:
        node_positions = nx.spring_layout(network, k=0.08)

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
    ax.set_title(graph_title, y=0.9)

    # Finally, draw the network graph
    nx.draw_networkx(
        network, pos=node_positions, node_color=scores, node_size=node_sizes, labels=node_labels,
        edge_color="silver", cmap=node_colormap, vmin=vmin, vmax=vmax, ax=ax, **networkx_kwds
    )

    # Add a colorbar to the axis
    sm = plt.cm.ScalarMappable(cmap=node_colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm)

    # Remove spines
    ax.axis('off')

    return ax


def draw_static_network_overview(
        network: nx.classes.graph.Graph,
        scores: Union[str, list],
        node_list: Union[None, list] = None,
        node_sizes: Union[None, str, list, dict] = None,
        node_labels: Union[None, str, list, dict] = None,
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
    score_column
        Name of the feature to show on this figure
    node_list
        List of nodes to show; useful if we want to prefilter based on score or pathway
    node_positions
        A mapping of node positions; fixing positions between samples can help comparison
    show_stats
        If the two extra plots should also be shown or just the network
    figsize
        Figure size; just to aim big, it defaults to a full-page A4 figure

    Returns
    -------
    A tuple of matplotlib axes.
    """

    # If some descriptive stats for scores and degrees are to be shown, it makes sense to access them now
    if show_stats:

        # Reconcile client inputs just as the static network drawer would do it, just a bit earlier
        network, node_list = check_node_feature_integrity(network, node_list)
        node_features = pd.DataFrame.from_dict(network.nodes, orient="index").reset_index()
        node_labels = get_node_labels(node_labels, node_features, node_list)
        scores = get_node_scores(scores, node_features, node_list)
        if node_sizes is None:
            node_sizes = dict(network.degree)
        node_sizes = get_node_sizes(node_sizes, node_features, node_list)

        fig, axs = plt.subplots(
            2, 3, figsize=figsize,
            gridspec_kw={"width_ratios": (1, 1, 1), "height_ratios": (4, 1)}
        )
        gs = axs[0, 0].get_gridspec()
        for ax in axs[0, 0:]:
            ax.remove()
        ax_net = fig.add_subplot(gs[0, 0:])

        draw_static_network(
            network, scores, node_list, node_sizes, node_labels, node_positions,
            node_size_factor, node_colormap, color_min_max, ax=ax_net
        )

        draw_netstat_histo(scores, label="Color", ax=axs[1, 0])
        draw_netstat_scatter(scores, node_sizes, labels=("Color", "Size"), ax=axs[1, 1])
        draw_netstat_histo(node_sizes, label="Size", ax=axs[1, 2])

        # If stats plots are not required, we can leave everything to the static network drawer
    else:
        fig, axs = plt.subplots(figsize=figsize)
        draw_static_network(
            network, scores, node_list, node_sizes, node_labels, node_positions,
            node_size_factor, node_colormap, color_min_max, ax=axs
        )

    fig.suptitle(main_title)
    plt.savefig("static_net.png")

    return axs
