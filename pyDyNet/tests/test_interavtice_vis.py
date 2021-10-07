import os


from ..visualization.interactive_vis import convert_input_data
from ..visualization.interactive_vis import draw_interactive_network

for filename in os.listdir("."):
    
    if filename.endswith(".html"):
        os.remove(filename)


def test_convert_input_data():

    nodetable = "pyDyNet/tests/vis_tests_files/nodetable.txt"
    edgetable = "pyDyNet/tests/vis_tests_files/edgetable.txt"
    results = convert_input_data(nodetable, edgetable)

    assert list(results.nodes) == [36, 50, 7, 49, 43, 47, 33, 31, 45, 44, 24, 19, 23, 17, 14, 42, 32, 28, 27, 11, 41, 34, 40, 35, 39, 16, 38, 30, 37, 21, 22, 12, 9, 29, 2, 10, 26, 25, 15, 8, 20, 48, 18, 4, 6, 13, 3, 46, 1, 5]


def test_draw_interactive_network():

    nodetable = "pyDyNet/tests/vis_tests_files/nodetable.txt"
    edgetable = "pyDyNet/tests/vis_tests_files/edgetable.txt"
    results = convert_input_data(nodetable, edgetable)

    html_results = draw_interactive_network(results, 800, 800, "output")
    html_counter = 0

    for filename in os.listdir("."):

        if filename.endswith("html"):
            html_counter += 1

    assert html_counter == 1
