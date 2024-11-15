# orchestration.py

import src.utils as utils
import src.main as train_and_test
import subprocess
import sys
import os


def install_requirements_from_file(requirements_file='requirements.txt'):
    """
    Installs all packages listed in the specified requirements.txt file using pip.

    Args:
        requirements_file (str): The path to the requirements.txt file (default is 'requirements.txt').
    """
    try:
        print(f"Installing packages from {requirements_file}...")
        # Run pip install command to install all the packages listed in the requirements file
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)


def setup(parameter=None):
    """
    Perform setup operations: generate requirements.txt and install dependencies.

    Args:
        parameter (str): Optional parameter to be passed to setup functions.
    """
    print(f"Running setup with parameter: {parameter}")

    directory_to_scan = '.'  # The directory to scan for Python files
    utils.generate_requirements_txt(directory_to_scan)  # Generate requirements.txt
    install_requirements_from_file()  # Install dependencies from requirements.txt


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
