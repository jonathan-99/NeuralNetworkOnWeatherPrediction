import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class NeuralNetworkModel:
    def __init__(self, input_shape, n_estimators=100, max_depth=5):
        """
        Initialize the model with input shape and hyperparameters.

        Args:
            input_shape (tuple): Shape of the input data (timesteps, features).
            n_estimators (int): The number of trees in the forest.
            max_depth (int): The maximum depth of the tree.
        """
        self.input_shape = input_shape
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = self._build_model()

    def _build_model(self):
        """
        Private method to build the regression model.

        Returns:
            model (RandomForestRegressor): Compiled machine learning model.
        """
        model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42)
        print("Model architecture successfully built.")
        return model

    def get_model_summary(self):
        """
        Print basic model information.
        """
        print(f"RandomForestRegressor Model with {self.n_estimators} estimators and max depth {self.max_depth}")

    def train(self, X_train, y_train):
        """
        Train the regression model.

        Args:
            X_train (np.array): Training feature data.
            y_train (np.array): Training target data.

        Returns:
            history (dict): Dictionary containing training metrics (mean squared error).
        """
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_train)
        mse = mean_squared_error(y_train, predictions)
        print(f"Training complete. Mean Squared Error on training data: {mse}")
        return {'mse': mse}

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data.

        Args:
            X_test (np.array): Test feature data.
            y_test (np.array): Test target data.

        Returns:
            loss (float): Loss value on the test data (MSE).
        """
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Test Mean Squared Error: {mse}")
        return mse

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

    # Simulate data (for example purposes)
    X_train = np.random.rand(100, *input_shape)
    y_train = np.random.rand(100, 1)
    X_test = np.random.rand(20, *input_shape)
    y_test = np.random.rand(20, 1)

    # Initialize the model
    model = NeuralNetworkModel(input_shape=input_shape, n_estimators=100, max_depth=5)
    model.get_model_summary()

    # Train the model
    model.train(X_train, y_train)

    # Evaluate the model
    model.evaluate(X_test, y_test)
