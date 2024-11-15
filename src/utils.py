# orchestration.py

import src.utils as utils
import src.main as train_and_test
import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def install_requirements_from_file(requirements_file='requirements.txt'):
    """
    Installs all packages listed in the specified requirements.txt file using pip.

    Args:
        requirements_file (str): The path to the requirements.txt file (default is 'requirements.txt').
    """
    try:
        logging.info(f"Installing packages from {requirements_file}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
        logging.info("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error installing packages: {e}")
        sys.exit(1)

def setup(parameter=None):
    """
    Perform setup operations: generate requirements.txt and install dependencies.

    Args:
        parameter (str): Optional parameter to be passed to setup functions.
    """
    logging.info(f"Running setup with parameter: {parameter}")

    directory_to_scan = '.'  # The directory to scan for Python files
    logging.info(f"Scanning directory {directory_to_scan} for Python files...")

    # Generate the requirements.txt file using utils
    utils.generate_requirements_txt(directory_to_scan)

    # Verify if the requirements.txt file is created
    if os.path.exists('requirements.txt'):
        logging.info("requirements.txt file created successfully.")
    else:
        logging.warning("requirements.txt file was not created.")

    # Install dependencies from the generated requirements.txt
    install_requirements_from_file()

def of_we_go(parameter=None):
    """
    Starts the neural network training process.

    Args:
        parameter (str): Optional parameter to control aspects of training.
    """
    logging.info(f"Running neural network training with parameter: {parameter}")
    train_and_test.main()  # Execute the main training process

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("Error: Missing parameter string. Please provide a parameter.")
        sys.exit(1)

    param_string = sys.argv[1]  # Get the first parameter from the command line argument
    setup(param_string)  # Step 1: Setup - Generate requirements.txt and install packages
    of_we_go(param_string)  # Step 2: Run the training process
