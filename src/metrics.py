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
            self.datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def to_dict(self):
            return {
                "filename": self.filename,
                "datetime": self.datetime
            }

    class ForestModel:
        def __init__(self):
            self.number_of_trees_in_forest = 0
            self.vc_dimension = 0
            self.rademacher_complexity = 0
            self.bayesian_information_criterion = 0

        def to_dict(self):
            return {
                "number_of_trees_in_forest": self.number_of_trees_in_forest,
                "vc_dimension": self.vc_dimension,
                "rademacher_complexity": self.rademacher_complexity,
                "bayesian_information_criterion": self.bayesian_information_criterion,
            }

    class NeuralModel:
        def __init__(self):
            self.temp = 0

        def to_dict(self):
            return {
                "temp": self.temp
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
                "regularisation_techniques": self.regularisation_techniques
            }

    class Timings:
        def __init__(self):
            self.loading_data = None
            self.preprocessing = None
            self.building_time_forest = None
            self.building_time_other = None
            self.training_time_forest = None
            self.training_time_other = None

        def to_dict(self):
            return {
                "loading_data": str(self.loading_data) if self.loading_data else None,
                "preprocessing": str(self.preprocessing) if self.preprocessing else None,
                "training_time_forest": str(self.building_time_forest) if self.building_time_forest else None,
                "training_time_other": str(self.building_time_other) if self.building_time_other else None,
                "training_time_forest": str(self.training_time_forest) if self.training_time_forest else None,
                "training_time_other": str(self.training_time_other) if self.training_time_other else None,
            }

    class Statistics:
        def __init__(self):
            self.validation_mse = 0.0
            self.training_mse = 0.0

        def to_dict(self):
            return {
                "validation_mse": self.validation_mse,
                "training_mse": self.training_mse,
            }

    class Accountant:
        def __init__(self):
            self.number_of_parameters = 0

        def to_dict(self):
            return {
                "number_of_parameters": self.number_of_parameters
            }

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.metadata = self.Metadata(self.timestamp)
        self.accountant = self.Accountant()
        self.statistics = self.Statistics()
        self.timings = self.Timings()
        self.hyperparameters = self.Hyperparameters()
        self.models = []  # List to store NeuralModel or ForestModel instances

    def add_model(self, model):
        """Add a model instance (either NeuralModel or ForestModel)."""
        if isinstance(model, (self.NeuralModel, self.ForestModel)):
            self.models.append(model)
        else:
            raise TypeError("Model must be an instance of NeuralModel or ForestModel")

    def get_metrics(self):
        return {
            "metadata": self.metadata.to_dict(),
            "accountant": self.accountant.to_dict(),
            "statistics": self.statistics.to_dict(),
            "timings": self.timings.to_dict(),
            "hyperparameters": self.hyperparameters.to_dict(),
            "models": [model.to_dict() for model in self.models]
        }

    def save_metric_to_file(self):
        results_dir = os.path.join(os.path.dirname(__file__), '../results')
        os.makedirs(results_dir, exist_ok=True)

        file_path = os.path.join(results_dir, f"{self.timestamp}.json")

        try:
            with open(file_path, 'w') as file:
                json.dump(self.get_metrics(), file, indent=4)
            logging.info(f"Metrics saved successfully to {file_path}")
        except IOError as e:
            logging.error(f"Error saving metrics to file: {e}")
            raise


# Example Usage
if __name__ == "__main__":
    metrics = Metrics()

    # Add models dynamically
    metrics.add_model(Metrics.NeuralModel())
    metrics.add_model(Metrics.ForestModel())

    # Save metrics
    metrics.save_metric_to_file()

    # Print metrics
    print(json.dumps(metrics.get_metrics(), indent=4))
