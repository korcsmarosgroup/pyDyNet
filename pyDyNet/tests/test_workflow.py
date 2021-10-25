import os
from distutils import dir_util
import numpy as np
import pandas as pd
import pytest
from pyDyNet.core.workflow import run_pydynet


# Expected outputs
EXPECTED_DN_RESULTS = np.array([[1.68872769, 5., 0.33774554],
                                [2.06286387, 5., 0.41257277],
                                [1.69652444, 5., 0.33930489],
                                [1.85747233, 5., 0.37149447],
                                [1.7800983, 5., 0.35601966]])


@pytest.fixture
def data_dir(tmpdir, request):
    """ A helper function for copying example and test data to the tmpdir """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)
    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))
    return tmpdir


def test_run_pydynet(data_dir):
    """ Test the pydynet workflow execution """
    output_dir = os.path.join(data_dir, "results")
    input_networks_directory = os.path.join(data_dir, "raw_data")
    # node_list_path = os.path.join(data_dir, "node_list", "node_list.txt")
    run_pydynet(input_networks_directory=input_networks_directory,
                network_format="adjacency",
                sep="\t",
                graph_type="undirected",
                weight_type="edge",
                output_dir=output_dir)
    actual_dn_results = pd.read_csv(os.path.join(output_dir, "dynet_results", "dn_scores.txt"), sep="\t")
    assert np.allclose(actual_dn_results.values, EXPECTED_DN_RESULTS)
