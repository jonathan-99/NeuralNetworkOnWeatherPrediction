import os
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor  # Example model
from sklearn.metrics import mean_squared_error


# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelTrainer:
    def __init__(self, model, output_dir='model_checkpoints'):
        """
        Initialize the ModelTrainer with the model and output directory.

        Args:
            model: The machine learning model (e.g., from sklearn).
            output_dir (str): Directory where the best model checkpoint will be saved.
        """
        self.model = model
        self.output_dir = output_dir

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logging.info(f"Created output directory at {self.output_dir}")

        self.checkpoint_path = os.path.join(self.output_dir, 'best_model.pkl')

    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """
        Train the model with training and validation data.

        Args:
            X_train (np.array): Training feature data.
            y_train (np.array): Training target data.
            X_val (np.array): Validation feature data.
            y_val (np.array): Validation target data.
            epochs (int): Number of epochs for training. Not used in this case, as scikit-learn models don't have epochs.

        Returns:
            history (dict): Dictionary containing training metrics (e.g., MSE).
        """
        best_mse = float('inf')
        logging.info("Starting model training...")

        try:
            # Train the model using the entire training data
            self.model.fit(X_train, y_train)
            logging.info("Model training completed.")

            # Evaluate on validation data
            val_predictions = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_predictions)
            logging.info(f"Validation MSE: {val_mse:.4f}")

            # Save the model if it's the best performing one
            if val_mse < best_mse:
                best_mse = val_mse
                joblib.dump(self.model, self.checkpoint_path)
                logging.info(f"Best model saved with MSE: {best_mse:.4f} at {self.checkpoint_path}")
            else:
                logging.info(f"Model with MSE: {val_mse:.4f} is not better than the best model.")

        except Exception as e:
            logging.error(f"An error occurred during training: {e}")

        return {
            'val_mse': val_mse,
            'best_mse': best_mse
        }

    def load_best_model(self):
        """
        Load the best model saved during training.
        """
        if os.path.exists(self.checkpoint_path):
            self.model = joblib.load(self.checkpoint_path)
            logging.info(f"Best model loaded from {self.checkpoint_path}")
        else:
            logging.error("Best model file not found.")


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Simulated training and validation data (replace with actual preprocessed data)
    input_shape = (10, 1)
    X_train = np.random.rand(100, *input_shape)
    y_train = np.random.rand(100, 1)
    X_val = np.random.rand(20, *input_shape)
    y_val = np.random.rand(20, 1)

    # Initialize a machine learning model (e.g., RandomForestRegressor from scikit-learn)
    model = RandomForestRegressor(n_estimators=100)

    # Train the model
    trainer = ModelTrainer(model)
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=50)

    # Load the best model for evaluation or prediction
    trainer.load_best_model()
