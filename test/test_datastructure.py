import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import PyDyNet
import numpy as np

def test_import1():
    networks = [
        np.array([[1, 1], [2, 1]]),
        np.array([[1, 1], [2, 1]]),
        np.array([[1, 1], [2, 1]]),
    ]
    pdn = PyDyNet.core.datastructure.PyDyNet.from_list_of_numpy_arrays(networks)

    assert len(pdn.InputNetworks) == 3
