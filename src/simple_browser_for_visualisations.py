# visualization_server.py

import os
import time
from flask import Flask, render_template_string, send_file
import matplotlib.pyplot as plt
import logging

# Set up logging to file or console
logging.basicConfig(level=logging.DEBUG)

# Create an app instance
app = Flask(__name__)

# Directory to store HTML visualizations
OUTPUT_DIR = 'visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of generated files
generated_files = []


# Function to create a visualization and save it as HTML
def create_visualization(timestamps, data, label, color, output_dir="visualizations"):
    """
    Creates and saves a plot of the data.

    Args:
        timestamps (list): The timestamps for the x-axis.
        data (list): The data to plot.
        label (str): The label for the data.
        color (str): The color of the plot line.
        output_dir (str): Directory to save the visualizations.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot the data
    plt.figure(figsize=(14, 6))
    plt.plot(timestamps, data, label=label, color=color)
    plt.xlabel('Timestamp')
    plt.ylabel('Wind Speed (m/s)')
    plt.title(f'{label} Wind Speed Data')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()

    # Save the plot as a PNG file with a timestamped name
    filename = os.path.join(output_dir, f'{label.replace(" ", "_")}_{timestamps[0].strftime("%Y-%m-%d_%H-%M-%S")}.png')
    plt.savefig(filename)
    plt.close()

    # Add the filename to the list of generated files for Flask to pick up
    generated_files.append(filename)  # Add this line

    return filename  # Return the file name to be used in the HTML


# Route to display available visualizations
@app.route('/')
def index():
    if not generated_files:
        return "No visualizations available. Please generate some first."

    file_links = ''.join(
        [f'<li><a href="/view/{os.path.basename(f)}">{os.path.basename(f)}</a></li>' for f in generated_files])
    return f"""
    <h1>Visualization Dashboard</h1>
    <p>Select a visualization to view:</p>
    <ul>{file_links}</ul>
    """


# Route to display a specific visualization
@app.route('/view/<filename>')
def view_file(filename):
    try:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            return "File not found."
        return send_file(filepath, mimetype='image/png')
    except Exception as e:
        logging.error(f"Error serving file {filename}: {e}")
        return "An error occurred while serving the file."



# Main function to generate visualizations
def main():
    from datetime import datetime
    import numpy as np

    # Sample data for testing
    timestamps = [datetime(2024, 2, 5, i) for i in range(24)]
    raw_data = np.random.rand(24) * 10  # Simulate raw wind speed data
    preprocessed_data = raw_data * 0.9  # Simulate scaled data

    # Create visualizations
    create_visualization(timestamps, raw_data, 'Raw Data', 'blue')
    create_visualization(timestamps, preprocessed_data, 'Preprocessed Data', 'green')

    # Run the web server
    app.run(host='0.0.0.0', port=2000)


if __name__ == '__main__':
    main()
