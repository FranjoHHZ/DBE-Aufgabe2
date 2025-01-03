import unittest
from ..src.Unit_download import download, split
from ..src.TheAlgorithm import TheAlgorithm

class TestRuntime(unittest.TestCase):
    def test_runtime(self):
        with open("data/baseline_runtime.txt", "r") as f:
            baseline_runtime = float(f.read().strip())
        threshold = 1.2 * baseline_runtime

        with open("data/runtime.txt", "r") as f:
            current_runtime = float(f.read().strip())

        self.assertLessEqual(current_runtime, threshold, "Runtime exceeded threshold!")
