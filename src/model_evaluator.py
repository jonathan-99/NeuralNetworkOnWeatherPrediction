"""
Description: Evaluates the trained model on the test set to measure performance metrics such as mean squared error (MSE) or root mean squared error (RMSE). Includes visualization of predictions vs. actual data.
"""

# model_evaluator.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf


class ModelEvaluator:
    def __init__(self, model):
        """
        Initialize the ModelEvaluator with the trained model.

        Args:
            model (tf.keras.Model): The trained neural network model.
        """
        self.model = model

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance on the test set.

        Args:
            X_test (np.array): Test feature data.
            y_test (np.array): True target values for the test data.

        Returns:
            dict: Dictionary containing evaluation metrics (MSE, RMSE, MAE).
        """
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)

        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

    def plot_predictions(self, y_test, predictions, timestamps=None):
        """
        Plot the actual vs. predicted values.

        Args:
            y_test (np.array): True target values for the test data.
            predictions (np.array): Model-predicted values for the test data.
            timestamps (list, optional): List of timestamps for the x-axis.
        """
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


# Example usage
if __name__ == "__main__":
    from model_trainer import ModelTrainer
    import numpy as np

    # Simulated test data (replace with actual test data)
    input_shape = (10, 1)
    X_test = np.random.rand(20, *input_shape)
    y_test = np.random.rand(20, 1)

    # Load the trained model
    model = tf.keras.models.load_model('model_checkpoints/best_model.h5')

    # Evaluate the model
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(X_test, y_test)

    # Generate predictions and plot
    predictions = model.predict(X_test)
    evaluator.plot_predictions(y_test, predictions)
