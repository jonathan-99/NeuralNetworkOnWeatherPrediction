"""
Description: Defines the architecture of the neural network using a deep learning framework (e.g., TensorFlow or PyTorch). Includes functions for compiling the model and setting up hyperparameters like learning rate and loss function.
"""

# model_definition.py

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


class NeuralNetworkModel:
    def __init__(self, input_shape, learning_rate=0.001):
        """
        Initialize the NeuralNetworkModel with input shape and hyperparameters.

        Args:
            input_shape (tuple): Shape of the input data (timesteps, features).
            learning_rate (float): Learning rate for the optimizer.
        """
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """
        Private method to build the neural network architecture.

        Returns:
            model (tf.keras.Model): Compiled neural network model.
        """
        model = Sequential()

        # LSTM layer to learn temporal dependencies
        model.add(LSTM(50, activation='relu', input_shape=self.input_shape, return_sequences=True))
        model.add(Dropout(0.2))

        # Second LSTM layer
        model.add(LSTM(50, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))

        # Fully connected output layer
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mean_squared_error')

        print("Model architecture successfully built and compiled.")
        return model

    def get_model_summary(self):
        """
        Print the summary of the model architecture.
        """
        self.model.summary()

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the neural network model.

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
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=2
        )
        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data.

        Args:
            X_test (np.array): Test feature data.
            y_test (np.array): Test target data.

        Returns:
            loss (float): Loss value on the test data.
        """
        loss = self.model.evaluate(X_test, y_test, verbose=2)
        print(f"Test loss: {loss}")
        return loss

    def predict(self, X):
        """
        Generate predictions using the trained model.

        Args:
            X (np.array): Feature data for predictions.

        Returns:
            np.array: Predicted values.
        """
        return self.model.predict(X)


# Example usage
if __name__ == "__main__":
    # Example input shape (e.g., time series data with 10 time steps and 1 feature)
    input_shape = (10, 1)  # Adjust as needed based on the preprocessed data
    model = NeuralNetworkModel(input_shape=input_shape, learning_rate=0.001)
    model.get_model_summary()
