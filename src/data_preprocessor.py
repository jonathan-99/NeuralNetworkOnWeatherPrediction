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
        logging.info(f"Original shape of wind_speeds: {self.wind_speeds.shape}")

        # Scale the data using Min-Max scaling
        self.wind_speeds = self.scaler.fit_transform(self.wind_speeds)

        # Log the min and max values of the scaled data
        logging.info(f"Data scaled with initial min: {np.min(self.wind_speeds)}, max: {np.max(self.wind_speeds)}.")

        logging.info(f"Scaled data shape: {self.wind_speeds.shape}")

        return self.wind_speeds

    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split the data into training, validation, and test sets.
        """
        logging.info(f"Splitting data: test_size={test_size}, val_size={val_size} (adjusted: {val_size_adjusted}).")
        train_val_data, test_data = train_test_split(
            self.wind_speeds, test_size=test_size, random_state=random_state, shuffle=False
        )

        logging.info(
            f"Train and test split: Train data shape = {train_val_data.shape}, Test data shape = {test_data.shape}")

        if len(train_val_data) == 0 or len(test_data) == 0:
            logging.error("Error: Training or test set is empty after split.")
            raise ValueError("Insufficient data to create non-empty splits.")

        val_size_adjusted = val_size / (1 - test_size)
        train_data, val_data = train_test_split(
            train_val_data, test_size=val_size_adjusted, random_state=random_state, shuffle=False
        )

        logging.info(f"Train, validation, and test split: Train data shape = {train_data.shape}, "
                     f"Validation data shape = {val_data.shape}, Test data shape = {test_data.shape}")

        return train_data, val_data, test_data

    def inverse_transform(self, data):
        """
        Inversely transform scaled data back to original scale.
        """
        logging.info(f"Inversely transforming data of shape {data.shape}.")

        return self.scaler.inverse_transform(data)
