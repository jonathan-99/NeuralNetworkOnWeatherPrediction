import src.capture_requirements as cap_req
import src.main as train_and_test
import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def is_package_installed(package_name):
    """
    Check if a package is installed using dpkg.

    Args:
        package_name (str): The name of the package to check.

    Returns:
        bool: True if the package is installed, False otherwise.
    """
    try:
        subprocess.run(['dpkg', '-s', package_name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def install_requirements_from_file(requirements_file='requirements.txt'):
    """
    Installs all packages listed in the specified requirements.txt file using apt-get if not already installed.

    Args:
        requirements_file (str): The path to the requirements.txt file (default is 'requirements.txt').
    """
    try:
        with open(requirements_file, 'r') as file:
            packages = file.readlines()

        for package in packages:
            package = package.strip()
            # Map Python package names to apt-get equivalents if needed
            apt_package = f"python3-{package}"  # This is an example; adjust based on your environment
            if not is_package_installed(apt_package):
                logging.info(f"Package {apt_package} is not installed. Installing...")
                subprocess.check_call(['sudo', 'apt-get', 'install', '-y', apt_package])
            else:
                logging.info(f"Package {apt_package} is already installed.")

        logging.info("Package installation process completed.")
    except FileNotFoundError:
        logging.error(f"{requirements_file} not found.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error installing package {package}: {e}")
        sys.exit(1)


def setup(parameter=None):
    """
    Perform setup operations: generate requirements.txt and check/install dependencies.

    Args:
        parameter (str): Optional parameter to be passed to setup functions.
    """
    logging.info(f"Running setup with parameter: {parameter}")

    directory_to_scan = '.'  # The directory to scan for Python files
    logging.info(f"Scanning directory {directory_to_scan} for Python files...")

    # Generate the requirements.txt file using utils
    cap_req.generate_requirements_txt(directory_to_scan)

    # Verify if the requirements.txt file is created
    if os.path.exists('requirements.txt'):
        logging.info("requirements.txt file found.")
        # Install dependencies from the generated requirements.txt
        install_requirements_from_file()
    else:
        logging.warning("requirements.txt file was not created.")


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
    setup(param_string)  # Step 1: Setup - Generate requirements.txt and check/install packages
    of_we_go(param_string)  # Step 2: Run the training process
