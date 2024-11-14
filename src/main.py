# main.py

from data_loader import load_data
from data_preprocessor import preprocess_data
from model_builder import build_model
from model_trainer import train_model
from model_evaluator import evaluate_model
from data_visualizer import plot_data, plot_predictions
from hyperparameter_tuner import tune_hyperparameters


def main():
    # Step 1: Load the data
    print("Loading data...")
    raw_data = load_data('path/to/wind_speed_data.txt')

    # Step 2: Preprocess the data
    print("Preprocessing data...")
    train_data, val_data, test_data = preprocess_data(raw_data)

    # Step 3: Visualize the preprocessed data (optional)
    print("Visualizing data...")
    plot_data(train_data)

    # Step 4: Build the model
    print("Building model...")
    model = build_model(input_shape=train_data.shape[1])

    # Step 5: Tune hyperparameters (optional step)
    # Uncomment if hyperparameter tuning is needed
    # print("Tuning hyperparameters...")
    # best_params = tune_hyperparameters(train_data, val_data)
    # model = build_model(input_shape=train_data.shape[1], **best_params)

    # Step 6: Train the model
    print("Training model...")
    history = train_model(model, train_data, val_data)

    # Step 7: Evaluate the model
    print("Evaluating model...")
    evaluate_model(model, test_data)

    # Step 8: Visualize predictions
    print("Visualizing predictions...")
    plot_predictions(model, test_data)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
