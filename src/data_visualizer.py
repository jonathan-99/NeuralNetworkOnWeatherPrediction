import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set(style="whitegrid")


class DataVisualizer:
    def __init__(self, timestamps, raw_data, preprocessed_data=None, model_predictions=None, output_dir="visualizations"):
        """
        Initialize the DataVisualizer with raw data, preprocessed data, and model predictions.

        Args:
            timestamps (list): List of datetime objects representing timestamps.
            raw_data (list): List of original wind speed data points.
            preprocessed_data (list, optional): List of scaled/normalized wind speed data points.
            model_predictions (list, optional): List of model-predicted wind speed data points.
            output_dir (str): Directory to save the visualizations.
        """
        self.timestamps = timestamps
        self.raw_data = raw_data
        self.preprocessed_data = preprocessed_data
        self.model_predictions = model_predictions
        self.output_dir = output_dir

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_plot(self, filename):
        """Save the current plot as a PNG file."""
        plt.savefig(filename, format='png')
        plt.close()

    def create_visualization(self, data, label, color):
        """Creates and saves a plot of the data."""
        plt.figure(figsize=(14, 6))
        plt.plot(self.timestamps, data, label=label, color=color)
        plt.xlabel('Timestamp')
        plt.ylabel('Wind Speed (m/s)')
        plt.title(f'{label} Wind Speed Data')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()

        # Save the plot as a PNG file
        filename = os.path.join(self.output_dir, f'{label.replace(" ", "_")}_{self.timestamps[0].strftime("%Y-%m-%d_%H-%M-%S")}.png')
        self.save_plot(filename)
        return filename  # Return the file name to be used in the HTML

    def generate_html(self, images, output_file="visualization_page.html"):
        """Generates an HTML page to display the saved images."""
        html_content = "<html><head><title>Wind Speed Visualizations</title></head><body>"

        # Add each image to the HTML file
        for image in images:
            html_content += f'<img src="{image}" alt="{image}" style="max-width: 100%; margin-bottom: 20px;"><br>'

        html_content += "</body></html>"

        # Save the HTML content to a file
        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"HTML file saved as {output_file}")

    def plot_raw_data(self):
        """Plot the raw wind speed data."""
        return self.create_visualization(self.raw_data, 'Raw Data', 'blue')

    def plot_preprocessed_data(self):
        """Plot the preprocessed wind speed data."""
        if self.preprocessed_data is None:
            print("No preprocessed data available to plot.")
            return
        return self.create_visualization(self.preprocessed_data, 'Preprocessed Data', 'green')

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

        # Save the plot as a PNG file
        filename = os.path.join(self.output_dir, f'Model_Predictions_vs_Raw_Data_{self.timestamps[0].strftime("%Y-%m-%d_%H-%M-%S")}.png')
        self.save_plot(filename)
        return filename

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

        # Save the plot as a PNG file
        filename = os.path.join(self.output_dir, f'Comparison_of_Raw_Preprocessed_and_Predicted_Data_{self.timestamps[0].strftime("%Y-%m-%d_%H-%M-%S")}.png')
        self.save_plot(filename)
        return filename


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

    # Create and save visualizations
    images = []
    images.append(visualizer.plot_raw_data())
    images.append(visualizer.plot_preprocessed_data())
    images.append(visualizer.plot_predictions())
    images.append(visualizer.plot_all())

    # Generate an HTML page with the visualizations
    visualizer.generate_html(images)
