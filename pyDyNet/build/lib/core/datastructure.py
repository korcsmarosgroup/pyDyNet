import networkx as nx
from parameters import PyDynetParameters

class PyDyNet(object):
    def __init__(self) -> None:
        self.InputNetworks = []
        self.Parameters = PyDynetParameters()
        self.Visualization = {}
        self.ResultNetwork = None
    
    @staticmethod
    def from_list_of_numpy_arrays(list_of_numpy_arrays):
        pdn = PyDyNet()
        for npa in list_of_numpy_arrays:
            pdn.InputNetworks.append(nx.from_numpy_array(npa, create_using=nx.MultiDiGraph))
        return pdn
    
    @staticmethod
    def from_list_of_numpy_matrixes(list_of_numpy_matrixes):
        pdn = PyDyNet()
        for npm in list_of_numpy_matrixes:
            pdn.InputNetworks.append(nx.from_numpy_matrix(npm, create_using=nx.MultiDiGraph))
        return pdn
