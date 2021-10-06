from pyDyNet import usecases
import argparse
import logging
import sys
import os


if os.path.isfile('pydynet.log'):
    os.remove("pydynet.log")

# Implement logging into the repository
logging.basicConfig(filename = 'pydynet.log', format = '%(levelname)s :: %(funcName)s :: %(lineno)d :: %(message)s', level = logging.INFO)


# Function for the input files and output files handling
def parse_args(args):

    help_text = \
        """
        === CLI for pyDyNet ===
        Basic cli python script, which handles the input and output file(s).

        **Parameters:** 
        
        -i, --input <path>                 : specified input file path [mandatory]

        -v, --visualisation <boolean>      : should the tool make visualisation or not [optional]
        
        -o, --output <path>                : specified output file path [mandatory]

        **Exit codes**

        Exit code 1: The specified input file does not exists!
        """

    parser = argparse.ArgumentParser(description = help_text)

    parser.add_argument("-i", "--input-file",
                        help="<path to the given input file> [mandatory]",
                        type=str,
                        dest="input_file",
                        action="store",
                        required=True)

    parser.add_argument("-v", "--vizualisation",
                        help="<should the tool make visualisation or not> [optional]",
                        dest="vizualisation",
                        action="store_true",
                        default=False)

    parser.add_argument("-o", "--output-file",
                        help="<path to the output file> [mandatory]",
                        type=str,
                        dest="output_file",
                        action="store",
                        required=True)

    results = parser.parse_args(args)
    return results.input_file, results.vizualisation, results.output_file


# Checking the input file is exists or not
def checking_input_file(input_file):

    if not os.path.isfile(input_file):
        sys.stderr.write(f"ERROR! the specified input file does not exists: {input_file}")
        logging.error(f"The specified input file does not exists: {input_file}! Exit code: 1")
        sys.exit(1)


# Main function for the CLI script
def main():

    input_file, vizualisation, output_file = parse_args(sys.argv[1:])

    checking_input_file(input_file)
    logging.info(f"Everything is fine on the CLI level! We are happy :)")

    usecases.dummy_use_case(input_file, output_file, "Yes")


if __name__ == "__main__":
    main()
