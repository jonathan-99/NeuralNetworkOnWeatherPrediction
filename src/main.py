# main.py

import logging
from src.data_loader import WindSpeedData
from src.data_preprocessor import DataPreprocessor
from src.data_visualizer import DataVisualizer
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.visualise_predictions import plot_predictions
from sklearn.ensemble import RandomForestRegressor  # Example model

# Configure logging for the main module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    try:
        # Step 1: Load the data
        logging.info("Step 1: Loading data...")
        wind_data = WindSpeedData(['train_data/2024-02-05.txt', 'train_data/2024-02-06.txt'])
        timestamps, wind_speeds = wind_data.get_data()

        if not timestamps or not wind_speeds:
            logging.error("Data loading failed. No data found.")
            return
        logging.info(f"Loaded data with {len(timestamps)} records.")

        # Step 2: Preprocess the data
        logging.info("Step 2: Preprocessing data...")
        preprocessor = DataPreprocessor(timestamps, wind_speeds)
        scaled_data = preprocessor.scale_data()

        # Get the data split into train, validation, and test sets
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()

        logging.info(f"Data split completed: "
                     f"Training set size: {X_train.shape[0]}, "
                     f"Validation set size: {X_val.shape[0]}, "
                     f"Test set size: {X_test.shape[0]}")

        if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
            logging.error("Data splitting resulted in an empty set. Check the input data and split parameters.")
            return

        # Step 3: Visualize the preprocessed data (optional)
        logging.info("Step 3: Visualizing raw and preprocessed data...")
        visualizer = DataVisualizer(timestamps, wind_speeds, preprocessed_data=scaled_data)
        visualizer.plot_raw_data()
        visualizer.plot_preprocessed_data()

        # Step 4: Build the model
        logging.info("Step 4: Building the model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)  # Replace with preferred sklearn model
        logging.info("Model initialized.")

        # Step 5: Train the model
        logging.info("Training model...")
        logging.info(f"Shape of train data: {X_train.shape}")
        logging.info(f"Shape of train features: {X_train.shape}")
        logging.info(f"Shape of train targets: {y_train.shape}")

        trainer = ModelTrainer(model)  # Initialize ModelTrainer with the model
        trainer.train(X_train, y_train)  # Pass the data and target

        logging.info("Model training completed.")

        # Step 6: Evaluate the model
        logging.info("Step 6: Evaluating the model...")
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X_test, y_test)

        logging.info(f"Evaluation metrics: {metrics}")

        # Generate predictions for visualization
        logging.info("Generating predictions for the test set...")
        predictions = model.predict(X_test)
        logging.info("Predictions generated.")

        # Visualize the predictions
        evaluator.plot_predictions(y_test, predictions)
        logging.info("Model evaluation and visualization completed.")

        # Step 7: Visualize predictions
        logging.info("Step 7: Visualizing final predictions...")
        plot_predictions(model, X_test, y_test)

        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"An unexpected error occurred during execution: {e}")


if __name__ == "__main__":
    main()
