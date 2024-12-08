import json
import logging
import os
from datetime import datetime

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Metrics:
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
                "regularisation_techniques": self.regularisation_techniques
            }

    class Timings:
        def __init__(self, loading_data=0.0, preprocessing=0.0, virtualising=0.0, building=0.0,
                     training=0.0, evaluating=0.0, prediction=0.0, visualising_predictions=0.0):
            self.loading_data = loading_data
            self.preprocessing = preprocessing
            self.virtualising = virtualising
            self.building = building
            self.training = training
            self.evaluating = evaluating
            self.prediction = prediction
            self.visualising_predictions = visualising_predictions

        def to_dict(self):
            return {
                "loading_data": self.loading_data,
                "preprocessing": self.preprocessing,
                "virtualising": self.virtualising,
                "building": self.building,
                "training": self.training,
                "evaluating": self.evaluating,
                "prediction": self.prediction,
                "visualising_predictions": self.visualising_predictions,
            }

    def __init__(self):
        self.validation_mse = 0.0
        self.training_mse = 0.0
        self.rmse_value = 0.0
        self.mae_value = 0.0
        self.val_mse = 0.0
        self.best_mse = 0.0
        self.number_of_layers = 0
        self.number_of_parameters = 0
        self.number_of_units_in_each_layer = []
        self.activation_functions = []
        self.vc_dimension = 0
        self.rademacher_complexity = 0.0
        self.bayesian_information_criterion = 0.0
        self.x_train_shape = 0.0
        self.y_train_shape = 0.0
        self.timings = self.Timings()
        self.hyperparameters = self.Hyperparameters()

    def set_validation_mse(self, value):
        if isinstance(value, (float, int)):
            self.validation_mse = float(value)
        else:
            logging.error("Invalid value for Validation MSE")
            raise ValueError("Validation MSE must be a float or integer.")

    # Similarly, add set_ methods with validation for other variables...

    def get_metrics(self):
        return {
            "validation_mse": self.validation_mse,
            "training_mse": self.training_mse,
            "rmse_value": self.rmse_value,
            "mae_value": self.mae_value,
            "number_of_layers": self.number_of_layers,
            "number_of_parameters": self.number_of_parameters,
            "number_of_units_in_each_layer": self.number_of_units_in_each_layer,
            "activation_functions": self.activation_functions,
            "vc_dimension": self.vc_dimension,
            "rademacher_complexity": self.rademacher_complexity,
            "bayesian_information_criterion": self.bayesian_information_criterion,
            "training_time": self.training_time,
            "hyperparameters": self.hyperparameters.to_dict()
        }

    def get_all(self):
        return self.get_metrics()

    def save_metric_to_file(self):
        # Ensure the 'results' directory exists
        results_dir = os.path.join(os.path.dirname(__file__), '../results')
        os.makedirs(results_dir, exist_ok=True)

        # Create a timestamped file name
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
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
        metrics.set_metric("number_of_layers", 3)
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
