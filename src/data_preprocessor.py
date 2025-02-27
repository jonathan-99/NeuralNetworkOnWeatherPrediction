import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataPreprocessor:
    def __init__(self, timestamps, wind_speeds):
        """
        Initialize the DataPreprocessor with timestamps and wind speed data.
        """
        if len(timestamps) == 0 or len(wind_speeds) == 0:
            logging.error("Error: Empty timestamps or wind speeds passed to DataPreprocessor.")
            raise ValueError("Timestamps or wind speeds cannot be empty.")

        logging.info("Initializing DataPreprocessor with valid data.")
        self.timestamps = timestamps
        self.wind_speeds = np.array(wind_speeds).reshape(-1, 1)  # Reshape for compatibility with scalers
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize scaler
        logging.info(f"DataPreprocessor initialized with {len(self.timestamps)} timestamps and wind speeds of shape {self.wind_speeds.shape}.")
        logging.info(f"Scaler initialized with feature range: {self.scaler.feature_range}.")

    def scale_data(self):
        """
        Scale the wind speed data using Min-Max normalization.
        """
        logging.info(f"   Original shape of wind_speeds: {self.wind_speeds.shape}")

        # Scale the data using Min-Max scaling
        self.wind_speeds = self.scaler.fit_transform(self.wind_speeds)

        # Log the min and max values of the scaled data
        logging.info(f"   Data scaled with initial min: {np.min(self.wind_speeds)}, max: {np.max(self.wind_speeds)}.")

        logging.info(f"   Scaled data shape: {self.wind_speeds.shape}")

        return self.wind_speeds

    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split the data into training, validation, and test sets.
        """
        if len(self.wind_speeds) == 0:
            logging.error("Error: Cannot split empty data.")
            raise ValueError("No data available for splitting.")

        logging.info("   Splitting data into training, validation, and test sets...")

        # First split into training+validation and test sets
        train_val_data, test_data = train_test_split(
            self.wind_speeds, test_size=test_size, random_state=random_state, shuffle=False
        )

        if len(train_val_data) == 0 or len(test_data) == 0:
            logging.error("Error: Training or test set is empty after split.")
            raise ValueError("Insufficient data to create non-empty splits.")

        # Adjust validation size relative to the training+validation data
        val_size_adjusted = val_size / (1 - test_size)

        # Log the adjusted validation size
        logging.info(f"   Adjusted validation size: {val_size_adjusted:.4f} (calculated based on test_size={test_size})")

        # Now split the training+validation set into separate train and validation sets
        train_data, val_data = train_test_split(
            train_val_data, test_size=val_size_adjusted, random_state=random_state, shuffle=False
        )

        # Ensure data is reshaped into feature and target sets (X, y)
        X_train = train_data  # Features for training
        y_train = train_data  # If the wind speed is the target, keep it as the target (or define your target)

        X_val = val_data  # Features for validation
        y_val = val_data  # Validation target

        X_test = test_data  # Features for test
        y_test = test_data  # Test target

        # Enhanced logging to show the shape of each dataset
        logging.info(f"   Data split completed:")
        logging.info(f"   Train set: Size = {len(X_train)}, Shape = {X_train.shape}, y_train shape = {y_train.shape}")
        logging.info(f"   Validation set: Size = {len(X_val)}, Shape = {X_val.shape}, y_val shape = {y_val.shape}")
        logging.info(f"   Test set: Size = {len(X_test)}, Shape = {X_test.shape}, y_test shape = {y_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def inverse_transform(self, data):
        """
        Inversely transform scaled data back to original scale.
        """
        logging.info(f"   Inversely transforming data of shape {data.shape}.")

        return self.scaler.inverse_transform(data)
