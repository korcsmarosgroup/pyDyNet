import pandas as pd
import networkx as nx
import numpy as np
from os import listdir
from typing import Tuple, List, Optional
from os.path import isfile, join, basename
from .exception_handler import UnImplementedFileFormat, IncompatibleDataType, UnknownWeightType, NetworkNotInitialized, \
    GraphTypeIncompatible
from ..analytics.dynet import create_union


class MultiNetworks:

    def __init__(self,
                 network_list: Optional[List[nx.classes.graph.Graph]] = None,
                 state_names: Optional[List[str]] = None) -> None:
        if network_list is None:
            network_list = []
        if state_names is None:
            state_names = []
        self.network_list = network_list
        self.state_names = state_names
        self.reference_network = None

    @staticmethod
    def file_iterator(dir_path: str) -> Tuple[List[str], List[str]]:
        """ Iterate over all matrices in directory """
        adj_file_list = [join(dir_path, file) for file in listdir(dir_path) if isfile(join(dir_path, file))]
        adj_state_names = [basename(name) for name in adj_file_list]
        return adj_file_list, adj_state_names

    def load_adjacency_matrix(self, dir_path: str,
                              sep: str = "\t",
                              graph_type: str = "undirected") -> None:
        """ Load adjacency matrix from directory """
        adj_file_list, adj_state_names = self.file_iterator(dir_path=dir_path)
        adj_matrices = [self._load_matrices(file_path=file, sep=sep) for file in adj_file_list]
        self.create_networks(network_data=adj_matrices, data_format="adjacency", graph_type=graph_type)
        self.state_names = adj_state_names

    def load_edge_list(self, dir_path: str,
                       sep: str = "\t",
                       graph_type: str = "undirected") -> None:
        """ Load edge list from directory """
        edge_file_list, edge_state_names = self.file_iterator(dir_path=dir_path)
        edge_lists = [self._load_edge_list(file_path=file, sep=sep) for file in edge_file_list]
        self.create_networks(network_data=edge_lists, data_format="edge_list", graph_type=graph_type)
        self.state_names = edge_state_names

    def create_reference_network(self) -> None:
        """ Create and store the union reference network """
        self.reference_network = create_union(self.network_list)

    def get_adjacency_matrices(self,
                               node_list: Optional[list] = None,
                               weight_type: Optional[str] = "edge") -> List[np.ndarray]:
        """ Returns a adjacency matrices/list of adjacency matrices from the network/network list """
        if self.network_list is not None:
            if weight_type == "edge":
                adj_matrices_list = [nx.to_numpy_array(i) for i in self.network_list]
                return adj_matrices_list
            elif weight_type == "node":
                adj_matrices_list = [np.where(nx.to_numpy_array(i, nodelist=node_list) > 0,  1, 0)
                                     for i in self.network_list]
                return adj_matrices_list
            else:
                raise UnknownWeightType()
        else:
            raise NetworkNotInitialized()

    def create_networks(self,
                        network_data: list,
                        data_format: str,
                        graph_type: str = "undirected",
                        source: str = "source",
                        target: str = "target",
                        edge_attr: str = "weight") -> None:
        """ Create networkx graph from adj matrices """
        data_type = type(network_data[0])
        assert all(isinstance(g, data_type) for g in network_data)

        if graph_type == "undirected":
            graph_type = nx.Graph
        elif graph_type == "directed":
            graph_type = nx.DiGraph
        else:
            raise GraphTypeIncompatible()

        if data_format == "adjacency":
            self._check_graph_type(adj_list=network_data, graph_type=graph_type)
            if data_type == np.ndarray:
                graph_list = [nx.from_numpy_array(g, create_using=graph_type) for g in network_data]
                self.network_list = graph_list
            elif data_type == pd.DataFrame:
                graph_list = [nx.from_pandas_adjacency(g, create_using=graph_type) for g in network_data]
                self.network_list = graph_list
            else:
                raise IncompatibleDataType("Please try with adjacency matrix.")
        elif data_format == "edge_list":
            if data_type == pd.DataFrame:
                graph_list = [nx.from_pandas_edgelist(g, source=source, target=target, edge_attr=edge_attr)
                              for g in network_data]
                self.network_list = graph_list
            else:
                raise IncompatibleDataType("Please try with edge list.")
        else:
            raise UnImplementedFileFormat("Please try with adjacency matrix or edge list.")

    @classmethod
    def _check_graph_type(cls, adj_list, graph_type):
        """ Check input matrices against the graph type """
        is_symmetric = np.array([cls.check_symmetric(s) for s in adj_list]).any()
        if graph_type is nx.Graph and not is_symmetric:
            raise GraphTypeIncompatible()

    @staticmethod
    def _load_edge_list(file_path: str,
                        sep: str = "\t") -> pd.DataFrame:
        """ Load a edge list from a file """
        return pd.read_csv(file_path, sep)

    @staticmethod
    def _load_matrices(file_path: str,
                       sep: str = "\t") -> pd.DataFrame:
        """ Load in a matrix """
        return pd.read_csv(file_path, sep=sep)
    
    @staticmethod
    def iter_pandas_to_array(matrix_df_list: list) -> list:
        """ Extract np array from pandas frames """
        return [frame.values for frame in matrix_df_list]

    @staticmethod
    def check_symmetric(mat: np.ndarray,
                        relative_tolerance: float = 1e-05,
                        absolute_tolerance: float = 1e-08) -> bool:
        return np.allclose(mat, mat.T, rtol=relative_tolerance, atol=absolute_tolerance)
