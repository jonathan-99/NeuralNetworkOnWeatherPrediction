import json
from datetime import timedelta
from pathlib import Path

class PerformanceAnalytics:
    def __init__(self, json_file_paths):
        """
        Initialize the PerformanceAnalytics class.

        :param json_file_paths: List of file paths to JSON files containing performance metrics.
        """
        self.json_file_paths = json_file_paths
        self.data = []
        self._load_data()

    def _load_data(self):
        """
        Load and parse the JSON files into the class.
        """
        for file_path in self.json_file_paths:
            try:
                with open(file_path, 'r') as f:
                    self.data.append(json.load(f))
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading {file_path}: {e}")

    def get_summary_statistics(self):
        """
        Calculate summary statistics for the validation MSE, training MSE, and MAE.

        :return: Dictionary of summary statistics.
        """
        summary = {
            "validation_mse": [],
            "training_mse": [],
            "mae_value": []
        }

        for record in self.data:
            stats = record.get("statistics", {})
            summary["validation_mse"].append(stats.get("validation_mse", 0))
            summary["training_mse"].append(stats.get("training_mse", 0))
            summary["mae_value"].append(stats.get("mae_value", 0))

        def calculate_avg(values):
            return sum(values) / len(values) if values else 0

        return {
            "average_validation_mse": calculate_avg(summary["validation_mse"]),
            "average_training_mse": calculate_avg(summary["training_mse"]),
            "average_mae_value": calculate_avg(summary["mae_value"])
        }

    def get_average_timings(self):
        """
        Calculate average timings for each stage.

        :return: Dictionary of average timings.
        """
        timings = {
            "loading_data": [],
            "preprocessing": [],
            "virtualising": [],
            "building": [],
            "evaluating": [],
            "prediction": [],
            "visualising_predictions": [],
            "training_time": []
        }

        for record in self.data:
            timing_data = record.get("timings", {})
            for key in timings:
                timing_str = timing_data.get(key, "0:00:00")
                timings[key].append(timedelta(seconds=self._parse_timedelta(timing_str).total_seconds()))

        def calculate_avg_deltas(deltas):
            total_seconds = sum(delta.total_seconds() for delta in deltas)
            return timedelta(seconds=total_seconds / len(deltas)) if deltas else timedelta(seconds=0)

        return {key: calculate_avg_deltas(values) for key, values in timings.items()}

    @staticmethod
    def _parse_timedelta(time_str):
        """
        Parse a time string in the format "HH:MM:SS" into a timedelta object.

        :param time_str: Time string to parse.
        :return: timedelta object.
        """
        try:
            parts = list(map(float, time_str.split(':')))
            return timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])
        except ValueError:
            return timedelta(seconds=0)

    def save_summary_to_file(self, output_path):
        """
        Save the summary statistics and average timings to a JSON file.

        :param output_path: Path to save the output JSON file.
        """
        summary = {
            "summary_statistics": self.get_summary_statistics(),
            "average_timings": self.get_average_timings()
        }
        try:
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=4, default=str)
            print(f"Summary saved to {output_path}")
        except IOError as e:
            print(f"Error saving summary to file: {e}")

# Example Usage
if __name__ == "__main__":
    json_files = ["data1.json", "data2.json", "data3.json"]  # Replace with actual paths
    analytics = PerformanceAnalytics(json_files)

    # Print summary statistics
    print("Summary Statistics:")
    print(analytics.get_summary_statistics())

    # Print average timings
    print("Average Timings:")
    print(analytics.get_average_timings())

    # Save summary to file
    analytics.save_summary_to_file("summary_output.json")
