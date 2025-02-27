import os
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.metrics import Metrics

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTrainer:
    def __init__(self, model=None, output_dir='model_checkpoints', test_size=0.2):
        """
        Initialize the ModelTrainer with a scikit-learn model, output directory, and test size for splitting.

        Args:
            model: The machine learning model (e.g., from sklearn). Defaults to RandomForestRegressor.
            output_dir (str): Directory where the best model checkpoint will be saved.
            test_size (float): Proportion of the data to be used as the validation set.
        """
        self.model = model if model is not None else RandomForestRegressor(n_estimators=100)
        self.output_dir = output_dir
        self.test_size = test_size

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logging.info(f"   Created output directory at {self.output_dir}")

        self.checkpoint_path = os.path.join(self.output_dir, 'best_model.pkl')

    def train(self, data, target):
        """
        Train the model with the provided dataset and log detailed training history.

        Args:
            data (np.array): Combined feature data.
            target (np.array): Target values for the data.

        Returns:
            dict: Dictionary containing training metrics (e.g., MSE).
        """
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=self.test_size, random_state=42)
        logging.info("   Starting model training...")
        logging.info(f"  Training data size: {X_train.shape[0]} samples, Validation data size: {X_val.shape[0]} samples")

        best_mse = float('inf')
        history = {}

        try:
            # Verify that the training features are non-empty
            if X_train.shape[1] == 0:
                logging.error("Training data has no features. Check the preprocessing step.")
                raise ValueError("Training data has no features.")

            # Train the model using the training data
            self.model.fit(X_train, y_train)
            logging.info("   Model training completed successfully.")

            # Evaluate on validation data
            val_predictions = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_predictions)
            logging.info(f"   Validation Mean Squared Error (MSE): {val_mse:.4f}")

            # Save the model if it's the best performing one
            if val_mse < best_mse:
                best_mse = val_mse
                joblib.dump(self.model, self.checkpoint_path)
                logging.info(f"   Best model saved with MSE: {best_mse:.4f} at {self.checkpoint_path}")
            else:
                logging.info(f"   Model with MSE: {val_mse:.4f} did not improve the best MSE: {best_mse:.4f}")

            # Add detailed information to the history dictionary
            history['val_mse'] = val_mse
            history['best_mse'] = best_mse
            history['train_samples'] = X_train.shape[0]
            history['val_samples'] = X_val.shape[0]

            logging.info("   Training history:")
            for key, value in history.items():
                logging.info(f"{key}: {value}")

        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise e

        return history

    def load_best_model(self):
        """
        Load the best model saved during training.
        """
        if os.path.exists(self.checkpoint_path):
            self.model = joblib.load(self.checkpoint_path)
            logging.info(f"   Best model loaded from {self.checkpoint_path}")
        else:
            logging.error("Best model file not found.")
