from src.data_loader import WindSpeedData
from src.data_preprocessor import DataPreprocessor
from src.data_visualizer import DataVisualizer
from src.model_builder import NeuralNetworkModel
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.visualise_predictions import plot_predictions
# from hyperparameter_tuner import tune_hyperparameters

def main():
    # Step 1: Load the data
    print("Loading data...")
    wind_data = WindSpeedData('train_data/2024-02-05.txt, train_data/2024-02-06.txt')
    timestamps, wind_speeds = wind_data.get_data()  # Ensure this method returns timestamps and wind speeds

    # Step 2: Preprocess the data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor(timestamps, wind_speeds)
    train_data, val_data, test_data = preprocessor.split_data()

    # Step 3: Visualize the preprocessed data (optional)
    print("Visualizing data...")
    DataVisualizer(train_data)

    # Step 4: Build the model
    print("Building model...")
    model = NeuralNetworkModel(input_shape=train_data.shape[1])

    # Step 5: Tune hyperparameters (optional step)
    # Uncomment if hyperparameter tuning is needed
    # print("Tuning hyperparameters...")
    # best_params = tune_hyperparameters(train_data, val_data)
    # model = build_model(input_shape=train_data.shape[1], **best_params)

    # Step 6: Train the model
    print("Training model...")
    history = ModelTrainer(model, train_data, val_data)

    # Step 7: Evaluate the model
    print("Evaluating model...")
    ModelEvaluator(model, test_data)

    # Step 8: Visualize predictions
    print("Visualizing predictions...")
    plot_predictions(model, test_data)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
