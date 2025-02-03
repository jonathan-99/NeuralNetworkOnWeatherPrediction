import json
import logging
import os
from datetime import datetime

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Metrics:
    class Metadata:
        def __init__(self, filename=None):
            self.filename = filename
            self.datetime = datetime.now()

        def to_dict(self):
            return {
                "filename": self.filename,
                "datetime": self.datetime
            }
    class Hyperparameters:
        def __init__(self, learning_rate=0.01, number_of_hidden_layers=1, batch_size=32, number_of_epochs=10,
                     regularisation_techniques=None):
            self.learning_rate = learning_rate
            self.number_of_hidden_layers = number_of_hidden_layers
            self.batch_size = batch_size
            self.number_of_epochs = number_of_epochs
            self.regularisation_techniques = regularisation_techniques or []

        def to_dict(self):
            return {
                "learning_rate": self.learning_rate,
                "number_of_hidden_layers": self.number_of_hidden_layers,
                "batch_size": self.batch_size,
                "number_of_epochs": self.number_of_epochs,
                "total_splits": self.total_splits,
                "regularisation_techniques": self.regularisation_techniques
            }

    class Timings:
        def __init__(self, loading_data=None, preprocessing=None, virtualising=None, building=None,
                     evaluating=None, prediction=None, visualising_predictions=None, training_time=None):
            """
            Initialize the Timings object.

            :param loading_data: Time spent loading data as a timedelta or None.
            :param preprocessing: Time spent preprocessing data as a timedelta or None.
            :param virtualising: Time spent virtualizing processes as a timedelta or None.
            :param building: Time spent building models as a timedelta or None.
            :param evaluating: Time spent evaluating models as a timedelta or None.
            :param prediction: Time spent on predictions as a timedelta or None.
            :param visualising_predictions: Time spent visualizing predictions as a timedelta or None.
            :param training_time: Time spent on training as a timedelta or None.
            """
            self.loading_data = loading_data
            self.preprocessing = preprocessing
            self.virtualising = virtualising
            self.building = building
            self.evaluating = evaluating
            self.prediction = prediction
            self.visualising_predictions = visualising_predictions
            self.training_time = training_time

        def to_dict(self):
            """
            Convert the Timings object to a dictionary, serializing timedelta objects to strings.
            """
            return {
                "loading_data": str(self.loading_data) if self.loading_data else None,
                "preprocessing": str(self.preprocessing) if self.preprocessing else None,
                "virtualising": str(self.virtualising) if self.virtualising else None,
                "building": str(self.building) if self.building else None,
                "evaluating": str(self.evaluating) if self.evaluating else None,
                "prediction": str(self.prediction) if self.prediction else None,
                "visualising_predictions": str(self.visualising_predictions) if self.visualising_predictions else None,
                "training_time": str(self.training_time) if self.training_time else None,
            }

        def set_timing(self, variable_name, start, finish):
            """
            Calculate and set the timing for a given variable as a timedelta.

            :param variable_name: The name of the attribute to set (e.g., 'loading_data').
            :param start: The start datetime.
            :param finish: The finish datetime.
            """
            if not hasattr(self, variable_name):
                raise AttributeError(f"{variable_name} is not a valid timing attribute.")

            if isinstance(start, datetime) and isinstance(finish, datetime):
                setattr(self, variable_name, finish - start)
            else:
                raise ValueError("Start and finish must be datetime objects.")

    class Statistics:
        def __init__(self):
            self.validation_mse = 0.0
            self.training_mse = 0.0
            self.rmse_value = 0.0
            self.mae_value = 0.0
            self.val_mse = 0.0
            self.best_mse = 0.0
            self.mse = 0.0
            self.total_splits = 0
            self.total_nodes = 0
            self.total_leaves = 0

        def to_dict(self):
            return {
                "validation_mse": self.validation_mse,
                "training_mse": self.training_mse,
                "rmse_value": self.rmse_value,
                "mae_value": self.mae_value,
                "mae_value": self.best_mse,
                "mse": self.mse,
                "total_splits": self.total_splits,
                "total_nodes": self.total_nodes,
                "total_leaves": self.total_leaves,
            }

    class Accountant:
        def __init__(self):
            self.number_of_parameters = 0
            self.number_of_units_in_each_layer = []
            self.activation_functions = []
            self.vc_dimension = 0
            self.rademacher_complexity = 0.0
            self.bayesian_information_criterion = 0.0
            self.x_train_shape = 0.0
            self.y_train_shape = 0.0
            self.max_depth = 0
            self.number_of_trees_in_forest = 0  # up == complexity

        def to_dict(self):
            return {
                "number_of_parameters": self.filename,
                "number_of_units_in_each_layer": self.datetime,
                "activation_functions": self.activation_functions,
                "vc_dimension": self.vc_dimension,
                "rademacher_complexity":  self.rademacher_complexity,
                "bayesian_information_criterion":self.bayesian_information_criterion,
                "x_train_shape": self.x_train_shape,
                "y_train_shape": self.y_train_shape,
                "max_depth": self.max_depth,
                "number_of_trees_in_forest": self.number_of_trees_in_forest
            }
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.Metadata = self.Metadata(self.timestamp)
        self.Accountant = self.Accountant()
        self.statistics = self.Statistics()
        self.timings = self.Timings()
        self.hyperparameters = self.Hyperparameters()

    def get_metrics(self):
        return {
            "statistics": self.statistics.to_dict(),
            "timings": self.timings.to_dict(),
            "hyperparameters": self.hyperparameters.to_dict(),
            "number_of_parameters": self.number_of_parameters,
            "number_of_units_in_each_layer": self.number_of_units_in_each_layer,
            "activation_functions": self.activation_functions,
            "vc_dimension": self.vc_dimension,
            "rademacher_complexity": self.rademacher_complexity,
            "bayesian_information_criterion": self.bayesian_information_criterion,
            "x_train_shape": self.x_train_shape,
            "y_train_shape": self.y_train_shape,
            "max_depth": self.max_depth,
            "number_of_trees_in_forest": self.number_of_trees_in_forest
        }

    def get_all(self):
        return self.get_metrics()

    def save_metric_to_file(self):
        # Ensure the 'results' directory exists
        results_dir = os.path.join(os.path.dirname(__file__), '../results')
        os.makedirs(results_dir, exist_ok=True)

        # Create a timestamped file name
        timestamp = self.timestamp
        file_path = os.path.join(results_dir, f"{timestamp}.json")

        # Check for existing file and avoid overwriting
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(results_dir, f"{timestamp}_{counter}.json")
            counter += 1

        # Save the metrics data to the file
        try:
            with open(file_path, 'w') as file:
                json.dump(self.get_all(), file, indent=4)
            logging.info(f"Metrics saved successfully to {file_path}")
        except IOError as e:
            logging.error(f"Error saving metrics to file: {e}")
            raise




# Example usage
if __name__ == "__main__":
    metrics = Metrics()
    try:
        # Set general metrics
        metrics.set_metric("Validation MSE", 0.01)
        metrics.set_metric("Training MSE", 0.008)
        metrics.set_metric("RMSE_value", 0.1)
        metrics.set_metric("MAE_value", 0.08)
        metrics.set_metric("number_of_parameters", 1500)
        metrics.set_metric("activation_functions", ["relu", "sigmoid"])
        metrics.set_metric("VC_dimension", 120)
        metrics.set_metric("Rademacher_complexity", 0.25)
        metrics.set_metric("Bayesian_Information_Criterion", -250.5)
        metrics.set_metric("training_time", 120.5)  # in seconds

        # Set hyperparameter metrics
        metrics.set_hyperparameter("Learning_Rate", 0.001)
        metrics.set_hyperparameter("number_of_hidden_layers", 2)
        metrics.set_hyperparameter("batch_size", 32)
        metrics.set_hyperparameter("number_of_epochs", 50)
        metrics.set_hyperparameter("Regularisation_Techniques", ["dropout", "L2"])

        # Retrieve and print individual metrics
        logging.info(f"Validation MSE: {metrics.get_metric('Validation MSE')}")
        logging.info(f"Learning Rate: {metrics.get_hyperparameter('Learning_Rate')}")

        # Print all metrics
        logging.info("All Metrics:")
        logging.info(metrics.get_all_metrics())

        # Save metrics to a file in the results folder
        metrics.save_metric_to_file()
        logging.info("Metrics saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
