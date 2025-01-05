import unittest
import numpy as np
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from TheAlgorithm import TheAlgorithm
from main import download_data, split

class TestAlgorithmRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('Setting up class for runtime tests.')

    def setUp(self):
        print('Setup for runtime test')
        X, y = download_data()
        splitRatio = 60000
        self.X_train, self.y_train, self.X_test, self.y_test = split(X, y, splitRatio)
        self.baseline_runtime = self.load_average_baseline_runtime()

    def load_average_baseline_runtime(self):
        data_dir = 'data'
        baseline_file_path = os.path.join(data_dir, 'baseline_runtime.txt')
        try:
            with open(baseline_file_path, 'r') as file:
                # Werte aus der Datei lesen, die durch ";" getrennt sind
                runtimes = file.read().strip().split(';')
                # Umwandeln der Werte in float und Berechnen des Durchschnitts
                average_runtime = np.mean([float(runtime) for runtime in runtimes if runtime])
                return average_runtime
        except FileNotFoundError:
            print(f"Datei {baseline_file_path} nicht gefunden.")
            return None

    def test_fit_runtime(self):
        if self.baseline_runtime is None:
            self.skipTest("Baseline-Laufzeit konnte nicht geladen werden.")
        
        np.random.seed(31337)
        ta = TheAlgorithm(self.X_train, self.y_train, self.X_test, self.y_test)

        # Start timing
        start_time = time.time()

        # Execute the fit() function
        ta.fit()

        # End timing
        end_time = time.time()

        # Calculate the actual duration
        actual_duration = end_time - start_time

        # Define expected maximum runtime = 120% of the average baseline runtime
        runtime_limit = self.baseline_runtime * 1.2  # 120% of the average baseline

        # Assert that the actual duration does not exceed the limit
        self.assertTrue(actual_duration <= runtime_limit, f"fit() function runtime exceeded the 120% limit of the baseline: {actual_duration} seconds; Limit was {runtime_limit} seconds")

if __name__ == '__main__':
    unittest.main()
