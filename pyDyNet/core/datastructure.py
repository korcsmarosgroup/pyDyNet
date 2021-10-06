import networkx as nx

class PydyNetwork(object):
    def __init__(self, network: nx.MultiDiGraph) -> None:
        self.Network = network
