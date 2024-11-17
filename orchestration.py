# orchestration.py

import src.utils as utils
import src.main as train_and_test
import subprocess
import sys
import os



def setup(parameter=None):
    """
    Perform setup operations: generate requirements.txt and install dependencies.

    Args:
        parameter (str): Optional parameter to be passed to setup functions.
    """
    print(f"Running setup with parameter: {parameter}")

    utils.setup()


def of_we_go(parameter=None):
    """
    Starts the neural network training process.

    Args:
        parameter (str): Optional parameter to control aspects of training.
    """
    print(f"Running neural network training with parameter: {parameter}")
    train_and_test.main()  # Execute the main training process


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing parameter string. Please provide a parameter.")
        sys.exit(1)

    param_string = sys.argv[1]  # Get the first parameter from the command line argument
    setup(param_string)  # Step 1: Setup - Generate requirements.txt and install packages
    of_we_go(param_string)  # Step 2: Run the training process
