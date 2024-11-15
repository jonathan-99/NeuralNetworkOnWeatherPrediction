"""
Description: Handles the training process of the neural network, including running the training loop, validating the model at each epoch, and saving the best model based on performance.
"""
# model_trainer.py

import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class ModelTrainer:
    def __init__(self, model, output_dir='model_checkpoints'):
        """
        Initialize the ModelTrainer with the neural network model and output directory.

        Args:
            model (tf.keras.Model): The compiled neural network model to be trained.
            output_dir (str): Directory where the best model checkpoint will be saved.
        """
        self.model = model
        self.output_dir = output_dir

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.checkpoint_path = os.path.join(self.output_dir, 'best_model.h5')

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model with training and validation data.

        Args:
            X_train (np.array): Training feature data.
            y_train (np.array): Training target data.
            X_val (np.array): Validation feature data.
            y_val (np.array): Validation target data.
            epochs (int): Number of epochs for training.
            batch_size (int): Size of the training batches.

        Returns:
            history (tf.keras.callbacks.History): History object with training metrics.
        """
        # Callbacks for saving the best model and early stopping
        checkpoint = ModelCheckpoint(
            filepath=self.checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=1
        )

        # Run the training process
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping],
            verbose=2
        )
        print(f"Training complete. Best model saved at {self.checkpoint_path}")
        return history

    def load_best_model(self):
        """
        Load the best model saved during training.
        """
        if os.path.exists(self.checkpoint_path):
            self.model = tf.keras.models.load_model(self.checkpoint_path)
            print(f"Best model loaded from {self.checkpoint_path}")
        else:
            print("Best model file not found.")


# Example usage
if __name__ == "__main__":
    from model_definition import NeuralNetworkModel
    import numpy as np

    # Simulated training and validation data (replace with actual preprocessed data)
    input_shape = (10, 1)
    X_train = np.random.rand(100, *input_shape)
    y_train = np.random.rand(100, 1)
    X_val = np.random.rand(20, *input_shape)
    y_val = np.random.rand(20, 1)

    # Initialize and compile the model
    model = NeuralNetworkModel(input_shape=input_shape).model

    # Train the model
    trainer = ModelTrainer(model)
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    # Load the best model for evaluation or prediction
    trainer.load_best_model()
