
"""
Description: Prepares the data for neural network training by scaling/normalizing, splitting it into training, validation, and test sets, and performing any necessary feature engineering.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self, timestamps, wind_speeds):
        """
        Initialize the DataPreprocessor with timestamps and wind speed data.

        Args:
            timestamps (list): List of datetime objects representing timestamps.
            wind_speeds (list): List of float values representing wind speed data.
        """
        self.timestamps = timestamps
        self.wind_speeds = np.array(wind_speeds).reshape(-1, 1)  # Reshape for compatibility with scalers
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize scaler

    def scale_data(self):
        """
        Scale the wind speed data using Min-Max normalization.

        Returns:
            np.array: Scaled wind speed data.
        """
        self.wind_speeds = self.scaler.fit_transform(self.wind_speeds)
        return self.wind_speeds

    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split the data into training, validation, and test sets.

        Args:
            test_size (float): Proportion of the data to use for testing.
            val_size (float): Proportion of the data to use for validation.
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple: Split data (train, validation, test) as numpy arrays.
        """
        # First, split into training + validation and test sets
        train_val_data, test_data = train_test_split(
            self.wind_speeds, test_size=test_size, random_state=random_state, shuffle=False
        )

        # Split the training + validation set into training and validation sets
        val_size_adjusted = val_size / (1 - test_size)
        train_data, val_data = train_test_split(
            train_val_data, test_size=val_size_adjusted, random_state=random_state, shuffle=False
        )

        return train_data, val_data, test_data

    def inverse_transform(self, data):
        """
        Inversely transform scaled data back to original scale.

        Args:
            data (np.array): Scaled data to be inversely transformed.

        Returns:
            np.array: Data in original scale.
        """
        return self.scaler.inverse_transform(data)


# Example usage
if __name__ == "__main__":
    from data_loader import WindSpeedData

    # Load data using the WindSpeedData class
    wind_data = WindSpeedData('path/to/wind_speed_data.txt')
    timestamps, wind_speeds = wind_data.get_data()

    # Process data using the DataPreprocessor class
    preprocessor = DataPreprocessor(timestamps, wind_speeds)

    # Scale data
    scaled_data = preprocessor.scale_data()
    print("Scaled data preview:", scaled_data[:5])

    # Split data
    train_data, val_data, test_data = preprocessor.split_data(test_size=0.3, val_size=0.2, random_state=40)
    print("Train set size:", len(train_data))
    print("Validation set size:", len(val_data))
    print("Test set size:", len(test_data))
