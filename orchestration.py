import argparse
import src.utils as utils
import src.main as train_and_test
from src.performance_analytics import PerformanceAnalytics

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

def analyse(json_file):
    """
    Analyzes the performance based on a provided JSON file.

    Args:
        json_file (str): Path to the JSON file to analyze.
    """
    print(f"Analyzing performance using file: {json_file}")
    analytics_object = PerformanceAnalytics(json_file)
    print("Summary Statistics:")
    print(analytics_object.get_summary_statistics())
    print("Average Timings:")
    print(analytics_object.get_average_timings())

def main():
    parser = argparse.ArgumentParser(description="Orchestrate neural network training and performance analysis.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for setup
    parser_setup = subparsers.add_parser("setup", help="Perform setup operations.")
    parser_setup.add_argument("parameter", type=str, nargs="?", help="Optional setup parameter.")

    # Subparser for training
    parser_train = subparsers.add_parser("train", help="Run the neural network training process.")
    parser_train.add_argument("parameter", type=str, nargs="?", help="Optional training parameter.")

    # Subparser for analysis
    parser_analyse = subparsers.add_parser("analyse", help="Analyze performance based on a JSON file.")
    parser_analyse.add_argument("json_file", type=str, help="Path to the JSON file for analysis.")

    args = parser.parse_args()

    if args.command == "setup":
        setup(args.parameter)
    elif args.command == "train":
        of_we_go(args.parameter)
    elif args.command == "analyse":
        analyse(args.json_file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()