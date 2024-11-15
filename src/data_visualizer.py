"""
Description: Generates plots and graphs to visualize the raw data, preprocessed data, and model predictions. This script can help provide insights into data trends and model behavior.
"""

# data_visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")


class DataVisualizer:
    def __init__(self, timestamps, raw_data, preprocessed_data=None, model_predictions=None):
        """
        Initialize the DataVisualizer with raw data, preprocessed data, and model predictions.

        Args:
            timestamps (list): List of datetime objects representing timestamps.
            raw_data (list): List of original wind speed data points.
            preprocessed_data (list, optional): List of scaled/normalized wind speed data points.
            model_predictions (list, optional): List of model-predicted wind speed data points.
        """
        self.timestamps = timestamps
        self.raw_data = raw_data
        self.preprocessed_data = preprocessed_data
        self.model_predictions = model_predictions

    def plot_raw_data(self):
        """Plot the raw wind speed data."""
        plt.figure(figsize=(14, 6))
        plt.plot(self.timestamps, self.raw_data, label='Raw Data', color='blue')
        plt.xlabel('Timestamp')
        plt.ylabel('Wind Speed (m/s)')
        plt.title('Raw Wind Speed Data')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_preprocessed_data(self):
        """Plot the preprocessed wind speed data."""
        if self.preprocessed_data is None:
            print("No preprocessed data available to plot.")
            return
        plt.figure(figsize=(14, 6))
        plt.plot(self.timestamps, self.preprocessed_data, label='Preprocessed Data', color='green')
        plt.xlabel('Timestamp')
        plt.ylabel('Scaled Wind Speed')
        plt.title('Preprocessed Wind Speed Data')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_predictions(self):
        """Plot the model predictions against the original data."""
        if self.model_predictions is None:
            print("No model predictions available to plot.")
            return
        plt.figure(figsize=(14, 6))
        plt.plot(self.timestamps, self.raw_data, label='Raw Data', color='blue', alpha=0.5)
        plt.plot(self.timestamps, self.model_predictions, label='Model Predictions', color='red')
        plt.xlabel('Timestamp')
        plt.ylabel('Wind Speed (m/s)')
        plt.title('Model Predictions vs. Raw Data')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_all(self):
        """Plot raw data, preprocessed data, and model predictions together."""
        plt.figure(figsize=(14, 8))
        plt.plot(self.timestamps, self.raw_data, label='Raw Data', color='blue', alpha=0.5)

        if self.preprocessed_data is not None:
            plt.plot(self.timestamps, self.preprocessed_data, label='Preprocessed Data', color='green', linestyle='--')

        if self.model_predictions is not None:
            plt.plot(self.timestamps, self.model_predictions, label='Model Predictions', color='red')

        plt.xlabel('Timestamp')
        plt.ylabel('Wind Speed (m/s)')
        plt.title('Comparison of Raw, Preprocessed, and Predicted Data')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.show()


# Example usage
if __name__ == "__main__":
    from data_loader import WindSpeedData
    from data_preprocessor import DataPreprocessor

    # Load and preprocess data
    wind_data = WindSpeedData('train_data/2024-02-05.txt, train_data/2024-02-06.txt')
    timestamps, raw_data = wind_data.get_data()

    preprocessor = DataPreprocessor(timestamps, raw_data)
    preprocessed_data = preprocessor.scale_data().flatten()

    # Simulate model predictions (for demonstration purposes)
    model_predictions = preprocessed_data * 0.9 + 0.1  # Example scaled prediction logic

    # Visualize data
    visualizer = DataVisualizer(timestamps, raw_data, preprocessed_data, model_predictions)
    visualizer.plot_raw_data()
    visualizer.plot_preprocessed_data()
    visualizer.plot_predictions()
    visualizer.plot_all()
