import os
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTrainer:
    def __init__(self, model, output_dir='model_checkpoints', test_size=0.2):
        self.model = model
        self.output_dir = output_dir
        self.test_size = test_size

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logging.info(f"Created output directory at {self.output_dir}")

        self.checkpoint_path = os.path.join(self.output_dir, 'best_model.pkl')

    def train(self, data, target, epochs=50):
        X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=self.test_size, random_state=42)
        logging.info("Starting model training...")

        best_mse = float('inf')
        val_mse = None  # Ensure val_mse is defined for return purposes

        try:
            # Check if the model has a fit method
            if not hasattr(self.model, 'fit'):
                raise AttributeError(f"'{type(self.model).__name__}' object has no attribute 'fit'")

            # Train the model
            self.model.fit(X_train, y_train)
            logging.info("Model training completed.")

            # Evaluate on validation data
            val_predictions = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_predictions)
            logging.info(f"Validation MSE: {val_mse:.4f}")

            if val_mse < best_mse:
                best_mse = val_mse
                joblib.dump(self.model, self.checkpoint_path)
                logging.info(f"Best model saved with MSE: {best_mse:.4f} at {self.checkpoint_path}")
            else:
                logging.info(f"Model with MSE: {val_mse:.4f} is not better than the best model.")

        except AttributeError as e:
            logging.error(f"Model error: {e}")
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")

        return {
            'val_mse': val_mse if val_mse is not None else float('inf'),
            'best_mse': best_mse
        }

    def load_best_model(self):
        if os.path.exists(self.checkpoint_path):
            self.model = joblib.load(self.checkpoint_path)
            logging.info(f"Best model loaded from {self.checkpoint_path}")
        else:
            logging.error("Best model file not found.")
