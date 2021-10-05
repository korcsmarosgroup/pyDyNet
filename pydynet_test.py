# Tests for the pydynet.py main script
import sys
import os


# Checking the input file is exists or not
def _checking_input_file(input_file):

    message = "Everything is ok!"
    if not os.path.isfile(input_file):
        message = "Something went wrong!"
    
    return message


def test_cheking_input_file():

    message = _checking_input_file("notexistingfile.txt")
    assert message == "Something went wrong!"
