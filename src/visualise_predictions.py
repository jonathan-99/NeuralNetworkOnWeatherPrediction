# visualize_predictions.py

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_predictions(model, X_test, y_test, timestamps=None):
    """
    Plot the actual vs. predicted values for the test data.

    Args:
        model (tf.keras.Model): Trained neural network model for prediction.
        X_test (np.array): Test feature data.
        y_test (np.array): True target values for the test data.
        timestamps (list, optional): List of timestamps for the x-axis.
    """
    # Generate predictions
    predictions = model.predict(X_test)

    # Create a plot
    plt.figure(figsize=(14, 6))
    if timestamps is None:
        timestamps = np.arange(len(y_test))  # Use indices if timestamps aren't provided

    plt.plot(timestamps, y_test, label='Actual Values', color='blue')
    plt.plot(timestamps, predictions, label='Predicted Values', color='red')
    plt.xlabel('Time')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Model Predictions vs. Actual Values')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load the trained model (ensure the path is correct)
    model = tf.keras.models.load_model('model_checkpoints/best_model.h5')

    # Simulated test data (replace with actual preprocessed test data)
    input_shape = (10, 1)  # Adjust to match your data shape
    X_test = np.random.rand(20, *input_shape)  # Example test features
    y_test = np.random.rand(20, 1)  # Example true target values

    # Visualize predictions
    print("Visualizing predictions...")
    plot_predictions(model, X_test, y_test)
