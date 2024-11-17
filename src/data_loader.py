import logging
from datetime import datetime

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WindSpeedData:
    def __init__(self, file_paths):
        """
        Initialize the WindSpeedData object by loading data from one or multiple text files.

        Args:
            file_paths (str or list): Path(s) to the text file(s) containing wind speed data.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]  # Convert a single file path to a list

        self.file_paths = file_paths
        self.timestamps, self.wind_speeds = self._load_data()

    def _load_data(self):
        """
        Private method to read wind speed data from the file(s) and store it with timestamps.

        Returns:
            tuple: Two lists - timestamps (datetime objects) and wind speed values (floats).
        """
        timestamps = []
        wind_speeds = []
        seen_timestamps = set()  # To track duplicates

        try:
            for file_path in self.file_paths:
                self._process_file(file_path, timestamps, wind_speeds, seen_timestamps)

            if not timestamps or not wind_speeds:
                logging.error("Loaded data is empty. Please check the file contents.")
            else:
                logging.info(f"Data loaded successfully from {len(self.file_paths)} file(s).")

        except FileNotFoundError:
            logging.error(f"Error: One or more files were not found: {self.file_paths}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while reading files: {e}")

        return timestamps, wind_speeds

    def _process_file(self, file_path, timestamps, wind_speeds, seen_timestamps):
        """
        Processes a single file to extract and validate wind speed data.

        Args:
            file_path (str): The path of the file to read.
            timestamps (list): The list to append timestamps.
            wind_speeds (list): The list to append wind speed data.
            seen_timestamps (set): The set to track unique timestamps.
        """
        logging.info(f"Reading data from {file_path}...")
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    # Ignore empty lines and process only valid lines
                    if line.strip():
                        self._process_line(line, timestamps, wind_speeds, seen_timestamps)
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")

    def _process_line(self, line, timestamps, wind_speeds, seen_timestamps):
        """
        Processes a single line of data, extracting and validating the timestamp and wind speed.

        Args:
            line (str): The line of text to process.
            timestamps (list): The list to append timestamps.
            wind_speeds (list): The list to append wind speed data.
            seen_timestamps (set): The set to track unique timestamps.
        """
        try:
            # Split the line by comma
            parts = line.strip().split(',')

            # Ensure the line splits into exactly 2 valid values (timestamp and wind speed)
            if len(parts) != 2:
                logging.warning(f"Skipping invalid data (wrong number of values): {line.strip()}")
                return

            timestamp_str, wind_speed_str = parts
            # Parse the timestamp and wind speed
            try:
                timestamp = datetime.strptime(timestamp_str.strip(), '%Y %m %d %H')
                wind_speed = float(wind_speed_str.strip())
            except ValueError as ve:
                logging.error(f"Skipping invalid data due to value error: {line.strip()} - {ve}")
                return

            # Check for duplicate empty rows (timestamps with 0.0)
            if timestamp in seen_timestamps and wind_speed == 0.0:
                logging.warning(f"Ignoring duplicate empty row for {timestamp}")
                return

            # Append parsed data to lists
            timestamps.append(timestamp)
            wind_speeds.append(wind_speed)
            seen_timestamps.add(timestamp)

        except Exception as e:
            logging.error(f"Skipping invalid data 3: {line.strip()} - {e}")

    def get_data(self):
        """
        Method to return the loaded timestamps and wind speed data.

        Returns:
            tuple: Two lists - timestamps and wind speed values.
        """
        return self.timestamps, self.wind_speeds


# Example usage
if __name__ == "__main__":
    # Replace 'path/to/wind_speed_data.txt' with actual paths or an array of file paths
    wind_data = WindSpeedData(['/train_data/2024-02-05.txt'])
    timestamps, wind_speeds = wind_data.get_data()

    # Display loaded data
    if timestamps and wind_speeds:
        for ts, ws in zip(timestamps, wind_speeds):
            logging.info(f"{ts}: {ws} m/s")
    else:
        logging.error("No valid data loaded.")
