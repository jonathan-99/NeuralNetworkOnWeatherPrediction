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

    def scale_data(self):
        """
        Scale the wind speed data using Min-Max normalization.
        """
        logging.info("Scaling data...")
        self.wind_speeds = self.scaler.fit_transform(self.wind_speeds)
        return self.wind_speeds

    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split the data into training, validation, and test sets.
        """
        if len(self.wind_speeds) == 0:
            logging.error("Error: Cannot split empty data.")
            raise ValueError("No data available for splitting.")

        logging.info("Splitting data into training, validation, and test sets...")
        train_val_data, test_data = train_test_split(
            self.wind_speeds, test_size=test_size, random_state=random_state, shuffle=False
        )

        if len(train_val_data) == 0 or len(test_data) == 0:
            logging.error("Error: Training or test set is empty after split.")
            raise ValueError("Insufficient data to create non-empty splits.")

        val_size_adjusted = val_size / (1 - test_size)
        train_data, val_data = train_test_split(
            train_val_data, test_size=val_size_adjusted, random_state=random_state, shuffle=False
        )

        logging.info(f"Data split completed: Train set size = {len(train_data)}, "
                     f"Validation set size = {len(val_data)}, Test set size = {len(test_data)}")

        return train_data, val_data, test_data

    def inverse_transform(self, data):
        """
        Inversely transform scaled data back to original scale.
        """
        logging.info("Inversely transforming scaled data.")
        return self.scaler.inverse_transform(data)
