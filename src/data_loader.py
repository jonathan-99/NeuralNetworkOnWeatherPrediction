# data_loader.py

from datetime import datetime


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
                with open(file_path, 'r') as file:
                    for line in file:
                        try:
                            # Split the line by comma
                            timestamp_str, wind_speed_str = line.strip().split(',')
                            # Parse the timestamp and wind speed
                            timestamp = datetime.strptime(timestamp_str, '%Y %m %d %H')
                            wind_speed = float(wind_speed_str)

                            # Check for duplicate empty rows (timestamps with 0.0)
                            if timestamp in seen_timestamps and wind_speed == 0.0:
                                print(f"Ignoring duplicate empty row for {timestamp}")
                                continue

                            # Append parsed data to lists
                            timestamps.append(timestamp)
                            wind_speeds.append(wind_speed)
                            seen_timestamps.add(timestamp)

                        except ValueError:
                            print(f"Skipping invalid data: {line.strip()}")
            print(f"Data loaded successfully from {len(self.file_paths)} file(s).")
        except FileNotFoundError:
            print(f"Error: One or more files were not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

        return timestamps, wind_speeds

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
    for ts, ws in zip(timestamps, wind_speeds):
        print(f"{ts}: {ws} m/s")
