import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add the parent directory to the sys.path to allow imports from the main project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ml_model import MLModel
from config import FEATURE_COLUMNS

class TestMLModel(unittest.TestCase):

    def setUp(self):
        """Set up a new MLModel instance before each test."""
        self.model = MLModel(algorithm='xgboost')

        # Create dummy data for testing
        self.X_train_np = np.random.rand(100, len(FEATURE_COLUMNS))
        self.y_train = np.random.randint(0, 2, 100)
        self.X_train = pd.DataFrame(self.X_train_np, columns=FEATURE_COLUMNS)

    def test_scaler_fitting(self):
        """Test that the StandardScaler is fitted after training."""
        # Train the model
        self.model.train(self.X_train, self.y_train)

        # Check if the scaler has been fitted
        self.assertIsNotNone(self.model.scaler.mean_)
        self.assertIsNotNone(self.model.scaler.scale_)
        self.assertEqual(len(self.model.scaler.mean_), len(FEATURE_COLUMNS))

    def test_scaler_transformation(self):
        """Test that the StandardScaler correctly transforms data."""
        # Train the model to fit the scaler
        self.model.train(self.X_train, self.y_train)

        # Create some test data
        X_test_np = np.random.rand(50, len(FEATURE_COLUMNS))
        X_test = pd.DataFrame(X_test_np, columns=FEATURE_COLUMNS)

        # Scale the test data using the predict method's internal scaling
        # We can't access the scaled data directly from predict, so we'll call the scaler manually
        X_test_scaled = self.model.scaler.transform(X_test)

        # The mean of the scaled data should be close to 0
        # This is not perfectly 0 because the scaler was fitted on the train set
        # But it should be reasonably close for a random dataset.
        self.assertTrue(np.all(np.abs(X_test_scaled.mean(axis=0)) < 1.0))

    def test_predict_runs(self):
        """Test that the predict method runs without errors."""
        self.model.train(self.X_train, self.y_train)
        X_test_np = np.random.rand(50, len(FEATURE_COLUMNS))
        X_test = pd.DataFrame(X_test_np, columns=FEATURE_COLUMNS)

        try:
            predictions = self.model.predict(X_test)
            self.assertEqual(len(predictions), 50)
        except Exception as e:
            self.fail(f"predict() raised an exception unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()
