import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib  # For loading models saved with joblib or sklearn
from src.metrics import metrics


class ModelEvaluator:
    def __init__(self, model):
        """
        Initialize the ModelEvaluator with the trained model.

        Args:
            model: The trained model (e.g., an sklearn model).
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
        # Make predictions
        predictions = self.model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)

        # Print the evaluation results
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
        # Plot the results
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
    import numpy as np

    # Simulated test data (replace with actual test data)
    input_shape = (10, 1)
    X_test = np.random.rand(20, *input_shape)
    y_test = np.random.rand(20, 1)

    # Load the trained model using joblib (assuming the model was saved as a .pkl file)
    model = joblib.load('model_checkpoints/best_model.pkl')  # Adjust if your model is in a different format

    # Evaluate the model
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(X_test, y_test)

    # Generate predictions and plot
    predictions = model.predict(X_test)
    evaluator.plot_predictions(y_test, predictions)
