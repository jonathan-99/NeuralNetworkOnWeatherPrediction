# main.py

import logging
from src.data_loader import WindSpeedData
from src.data_preprocessor import DataPreprocessor
from src.data_visualizer import DataVisualizer
from src.model_builder import NeuralNetworkModel
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.visualise_predictions import plot_predictions

# Configure logging for the main module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Step 1: Load the data
        logging.info("Loading data...")
        wind_data = WindSpeedData(['train_data/2024-02-05.txt', 'train_data/2024-02-06.txt'])
        timestamps, wind_speeds = wind_data.get_data()

        if not timestamps or not wind_speeds:
            logging.error("Data loading failed. No data found.")
            return

        # Step 2: Preprocess the data
        logging.info("Preprocessing data...")
        preprocessor = DataPreprocessor(timestamps, wind_speeds)
        scaled_data = preprocessor.scale_data()
        train_data, val_data, test_data = preprocessor.split_data()

        if train_data.size == 0 or val_data.size == 0 or test_data.size == 0:
            logging.error("Data splitting resulted in an empty set. Check the input data and split parameters.")
            return

        # Step 3: Visualize the preprocessed data (optional)
        logging.info("Visualizing data...")
        visualizer = DataVisualizer(timestamps, wind_speeds, preprocessed_data=scaled_data)
        visualizer.plot_raw_data()
        visualizer.plot_preprocessed_data()

        # Step 4: Build the model
        logging.info("Building model...")
        model = NeuralNetworkModel(input_shape=train_data.shape[1])

        # Step 5: Train the model
        logging.info("Training model...")
        trainer = ModelTrainer(model)  # Initialize ModelTrainer with the model
        history = trainer.train(train_data[:, :-1], train_data[:, -1])  # Pass the data and target

        # Step 6: Evaluate the model
        # Split test_data into features (X_test) and targets (y_test)
        X_test, y_test = test_data[:, :-1], test_data[:, -1]  # Last column as target

        # Initialize the ModelEvaluator with the trained model
        evaluator = ModelEvaluator(model)
        # Evaluate the model's performance
        metrics = evaluator.evaluate(X_test, y_test)

        # Generate predictions
        predictions = model.predict(X_test)

        # Visualize the predictions
        evaluator.plot_predictions(y_test, predictions)

        logging.info("Model evaluation completed.")

        # Step 7: Visualize predictions
        logging.info("Visualizing predictions...")
        plot_predictions(model, test_data)

        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"An unexpected error occurred during execution: {e}")

if __name__ == "__main__":
    main()
