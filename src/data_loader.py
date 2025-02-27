import logging
from datetime import datetime

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WindSpeedData:
    def __init__(self, file_paths):
        """
        Initialize the WindSpeedData object by loading data from one or multiple text files.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]  # Convert a single file path to a list

        self.file_paths = file_paths
        self.timestamps, self.wind_speeds = self._load_data()

    def _load_data(self):
        """
        Private method to read wind speed data from the file(s) and store it with timestamps.
        """
        timestamps = []
        wind_speeds = []
        seen_timestamps = set()  # To track duplicates

        try:
            for file_path in self.file_paths:
                self._process_file(file_path, timestamps, wind_speeds, seen_timestamps)

            if not timestamps or not wind_speeds:
                logging.error("   Loaded data is empty. Please check the file contents.")
            else:
                logging.info(f"   Data loaded successfully from {len(self.file_paths)} file(s).")

        except FileNotFoundError:
            logging.error(f"Error: One or more files were not found: {self.file_paths}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while reading files: {e}")

        return timestamps, wind_speeds

    def _process_file(self, file_path, timestamps, wind_speeds, seen_timestamps):
        """
        Processes a single file to extract and validate wind speed data.
        """
        logging.info(f"  Reading data from {file_path}...")
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    # Ignore empty lines and process only valid lines
                    if line.strip():
                        self._process_line(line, timestamps, wind_speeds, seen_timestamps)
        except FileNotFoundError:
            logging.error(f"  File not found: {file_path}")
        except PermissionError:
            logging.error(f"  Permission denied: {file_path}")
        except Exception as e:
            logging.error(f"  Unexpected error reading {file_path}: {e}")

    def _process_line(self, line, timestamps, wind_speeds, seen_timestamps):
        """
        Processes a single line of data, extracting and validating the timestamp and wind speed.
        """
        try:
            logging.debug(f"Processing line: {line.strip()}")  # Log raw line input

            # Split the line by comma and remove empty trailing parts
            parts = [x.strip() for x in line.strip().split(',') if x.strip()]
            logging.debug(f"Extracted parts: {parts}")  # Log extracted values

            # Ensure the line splits into exactly 2 valid values (timestamp and wind speed)
            if len(parts) != 2:
                logging.warning(f"   Skipping invalid data (wrong number of values): {line.strip()}")
                return

            timestamp_str, wind_speed_str = parts
            logging.debug(f"Parsed timestamp string: {timestamp_str}, Wind speed string: {wind_speed_str}")

            # Parse the timestamp and wind speed
            try:
                timestamp = datetime.strptime(timestamp_str.strip(), '%Y %m %d %H')
                wind_speed = float(wind_speed_str.strip())
                logging.debug(f"Converted timestamp: {timestamp}, Wind speed: {wind_speed}")
            except ValueError as ve:
                logging.error(f"   Skipping invalid data due to value error: {line.strip()} - {ve}")
                return

            # Check for duplicate empty rows (timestamps with 0.0)
            if timestamp in seen_timestamps and wind_speed == 0.0:
                logging.warning(f"   Ignoring duplicate empty row for {timestamp}")
                return

            # Append parsed data to lists
            timestamps.append(timestamp)
            wind_speeds.append(wind_speed)
            seen_timestamps.add(timestamp)
            logging.debug(f"Added data - Timestamp: {timestamp}, Wind Speed: {wind_speed}")

        except Exception as e:
            logging.error(f"   Skipping invalid data due to unexpected error: {line.strip()} - {e}")

    def get_data(self):
        """
        Method to return the loaded timestamps and wind speed data.
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
