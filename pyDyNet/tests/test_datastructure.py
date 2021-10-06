from ..core.datastructure import PyDyNet

def test_import1():
    networks = [
        np.array([[1, 1], [2, 1]]),
        np.array([[1, 1], [2, 1]]),
        np.array([[1, 1], [2, 1]]),
    ]
    pdn = PyDyNet.core.datastructure.PyDyNet.from_list_of_numpy_arrays(networks)

    assert len(pdn.InputNetworks) == 3
