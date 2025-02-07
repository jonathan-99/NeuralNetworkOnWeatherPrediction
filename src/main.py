# main.py

import logging
from src.data_loader import WindSpeedData
from src.data_preprocessor import DataPreprocessor
from src.data_visualizer import DataVisualizer
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.visualise_predictions import plot_predictions
from src.model_builder import ForestModel
from src.metrics import Metrics
import datetime

# Configure logging for the main module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_metrics(temp_object:Metrics):
    temp_object.ForestModel.number_of_trees_in_forest = 100
    temp_object.ForestModel.max_depth = 5

    return temp_object

def main():
    try:
        metric_object_initial = Metrics()
        metric_object = setup_metrics(metric_object_initial)

        # Step 1: Load the data
        start = datetime.datetime.now()
        logging.info("Step 1: Loading data...")

        wind_data = WindSpeedData(['train_data/2024-02-05.txt',
                                   'train_data/2024-02-06.txt',
                                   'train_data/2024-02-07.txt'])
        timestamps, wind_speeds = wind_data.get_data()
        if not timestamps or not wind_speeds:
            logging.error("   Data loading failed. No data found.")
            return
        logging.info(f"   Loaded data with {len(timestamps)} records.")

        finish = datetime.datetime.now()
        metric_object.timings.loading_data = finish - start



        # Step 2: Preprocess the data
        start = datetime.datetime.now()
        logging.info("Step 2: Preprocessing data...")

        preprocessor = DataPreprocessor(timestamps, wind_speeds)
        scaled_data = preprocessor.scale_data()
        # Get the data split into train, validation, and test sets
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
        logging.info(f"   Data split completed: "
                     f"   Training set size: {X_train.shape[0]}, "
                     f"   Validation set size: {X_val.shape[0]}, "
                     f"   Test set size: {X_test.shape[0]}")

        if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
            logging.error("   Data splitting resulted in an empty set. Check the input data and split parameters.")
            return

        finish = datetime.datetime.now()
        metric_object.timings.preprocessing = finish - start

        # Step 3: Visualize the preprocessed data (optional)
        start = datetime.datetime.now()
        logging.info("Step 3: Visualizing raw and preprocessed data...")

        visualizer = DataVisualizer(timestamps, wind_speeds, preprocessed_data=scaled_data)
        visualizer.plot_raw_data()
        visualizer.plot_preprocessed_data()

        finish = datetime.datetime.now()
        metric_object.timings.virtualising = finish - start

        # Step 4: Build the model
        start = datetime.datetime.now()
        logging.info("Step 4: Building the model...")

        # Define input shape based on training data
        input_shape = X_train.shape[1:]  # this assumes the correct format...error check
        # Initialize the model using NeuralNetworkModel from model_builder.py
        forest_model = ForestModel(input_shape=input_shape,
                                   n_estimators=metric_object.models.ForestModel.number_of_trees_in_forest,
                                   max_depth=metric_object.models.ForestModel.max_depth)
        y_train = y_train.ravel()  # or use y_train.flatten() to make sure it's 1D

        finish = datetime.datetime.now()
        metric_object.timings.building = finish - start

        # Step 5: Train the model
        start = datetime.datetime.now()
        logging.info("Step 5: Training model...")
        metric_object.x_train_shape = X_train.shape
        metric_object.y_train_shape = y_train.shape

        # Initialize ModelTrainer and train the model
        trainer = ModelTrainer(forest_model)
        training_output = trainer.train(X_train, y_train)

        # Capture training metrics
        train_metrics = forest_model.train(X_train, y_train)
        advanced_metrics = forest_model.get_advanced_metrics(X_train, y_train)
        model_structure_metrics = forest_model.get_model_structure_metrics()

        # Add model to metrics object
        metric_object.add_model(Metrics.ForestModel())

        # Store advanced model insights
        metric_object.ForestModel.vc_dimension = model_structure_metrics["vc_dimension"]
        metric_object.ForestModel.rademacher_complexity = model_structure_metrics["rademacher_complexity"]
        metric_object.ForestModel.bayesian_information_criterion = model_structure_metrics[
            "bayesian_information_criterion"]

        # Store model statistics
        metric_object.statistics.total_splits = train_metrics['total_splits']
        metric_object.statistics.total_nodes = train_metrics['total_nodes']
        metric_object.statistics.total_leaves = train_metrics['total_leaves']
        metric_object.statistics.training_mse = train_metrics['mse']
        metric_object.statistics.validation_mse = training_output['val_mse']
        metric_object.statistics.best_mse = training_output['best_mse']

        # Store additional advanced metrics
        metric_object.statistics.vc_dimension = advanced_metrics['vc_dimension']
        metric_object.statistics.rademacher_complexity = advanced_metrics['rademacher_complexity']
        metric_object.statistics.bayesian_information_criterion = advanced_metrics['bayesian_information_criterion']

        # Log results
        logging.info(f"   *Model structure metrics: {model_structure_metrics}")
        logging.info(f"   *Advanced Metrics: VC Dimension={advanced_metrics['vc_dimension']}, "
                     f"   Rademacher Complexity={advanced_metrics['rademacher_complexity']}, "
                     f"   BIC={advanced_metrics['bayesian_information_criterion']}")
        logging.info(f"   Training history - {training_output}")
        logging.info("   Forest Model training completed.")

        finish = datetime.datetime.now()
        metric_object.timings.training_time = finish - start

        # Step 6: Evaluate the model
        start = datetime.datetime.now()
        logging.info("Step 6: Evaluating the model...")
        evaluator = ModelEvaluator(forest_model)
        metrics = evaluator.evaluate(X_test, y_test)
        metric_object.mse = metrics['mse']
        metric_object.rmse_value = metrics['rmse']
        metric_object.mae_value = metrics['mae']
        finish = datetime.datetime.now()
        metric_object.timings.evaluating = finish - start
        logging.info(f"   Evaluation metrics: {metrics}")

        # Generate predictions for visualization
        start = datetime.datetime.now()
        logging.info("    Generating predictions for the test set...")
        predictions = forest_model.predict(X_test)
        finish = datetime.datetime.now()
        metric_object.timings.prediction = finish - start
        logging.info("   Predictions generated.")

        # Visualize the predictions
        start = datetime.datetime.now()
        evaluator.plot_predictions(y_test, predictions)
        finish = datetime.datetime.now()
        metric_object.timings.visualising_predictions = finish - start
        logging.info("   Model evaluation and visualization completed.")

        # Step 7: Visualize predictions
        logging.info("Step 7: Visualizing final predictions...")
        start = datetime.datetime.now()
        plot_predictions(forest_model, X_test, y_test)
        finish = datetime.datetime.now()
        metric_object.timings.visualising = finish - start

        logging.info("   Pipeline completed successfully.")
        metric_object.save_metric_to_file()
    except Exception as e:
        logging.error(f"An unexpected error occurred during execution: {e}")


if __name__ == "__main__":
    main()
