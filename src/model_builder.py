import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class ForestModel:
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
        logging.info(f"   Model architecture successfully built.")
        return model

    def get_model_structure_metrics(self):
        """
        Retrieve the splits, nodes, and leaves for the entire forest.

        Returns:
            dict: A dictionary with total splits, total nodes, and total leaves.
        """
        total_splits = 0
        total_nodes = 0
        total_leaves = 0

        for tree in self.model.estimators_:
            tree_structure = tree.tree_
            total_nodes += tree_structure.node_count
            total_leaves += tree_structure.n_leaves
            total_splits += tree_structure.node_count - tree_structure.n_leaves

        return {
            "total_splits": total_splits,
            "total_nodes": total_nodes,
            "total_leaves": total_leaves,
        }

    def get_advanced_metrics(self, X_train, y_train):
        if not hasattr(self.model, "estimators_"):
            raise ValueError("Model must be trained before calculating advanced metrics.")

        metrics = self.get_model_structure_metrics()
        total_splits = metrics["total_splits"]
        total_nodes = metrics["total_nodes"]

        # VC Dimension: Approximation
        vc_dimension = total_splits

        # Rademacher Complexity: Proxy using noise fitting
        random_noise = np.random.choice([-1, 1], size=len(y_train))  # Random binary noise
        self.model.fit(X_train, random_noise)
        noise_predictions = self.model.predict(X_train)
        noise_fit_mse = mean_squared_error(random_noise, noise_predictions)
        rademacher_complexity = noise_fit_mse  # Proxy: Lower MSE -> Higher complexity

        # BIC: Using training MSE
        self.model.fit(X_train, y_train)  # Re-train on actual data
        predictions = self.model.predict(X_train)
        mse = mean_squared_error(y_train, predictions)
        n_samples = len(X_train)
        bic = n_samples * np.log(mse) + total_nodes * np.log(n_samples)

        return {
            "vc_dimension": vc_dimension,
            "rademacher_complexity": rademacher_complexity,
            "bayesian_information_criterion": bic,
        }

    def get_model_summary(self):
        """
        Print basic model information and structure metrics.
        """
        text = f"   RandomForestRegressor Model with {self.n_estimators} estimators and max depth {self.max_depth}"
        logging.info(text)
        print(text)
        if hasattr(self.model, "estimators_"):
            metrics = self.get_model_structure_metrics()
            print("   Model Structure Metrics:")
            print(f"  Total Nodes: {metrics['total_nodes']}")
            print(f"  Total Leaves: {metrics['total_leaves']}")
            print(f"  Total Splits: {metrics['total_splits']}")
        else:
            print("   Model structure metrics are not available until the model is trained.")

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
        logging.info(f"   Training complete. Mean Squared Error on training data: {mse}")
        return {'mse': mse}

    def fit(self, X_train, y_train):
        """
        Alias for train() to maintain compatibility with external calls.
        """
        return self.train(X_train, y_train)

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
        print(f"   Test Mean Squared Error: {mse}")
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
    y_train = np.random.rand(100, 1).ravel()
    X_test = np.random.rand(20, *input_shape)
    y_test = np.random.rand(20, 1).flatten()

    # Initialize the model
    model = ForestModel(input_shape=input_shape, n_estimators=100, max_depth=5)

    # Train the model
    model.train(X_train, y_train)

    # Print model summary including structure metrics
    model.get_model_summary()

    # Evaluate the model
    model.evaluate(X_test, y_test)
