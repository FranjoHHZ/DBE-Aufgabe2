import unittest
from src.Unit_download import download, split
from src.TheAlgorithm import TheAlgorithm

class TestFit(unittest.TestCase):
    def test_fit_function(self):
        X, y = download()
        X_train, y_train, X_test, y_test = split(X, y, split_ratio=60000)
        model = TheAlgorithm(X_train, y_train, X_test, y_test)
        model.fit()

        self.assertTrue(hasattr(model.model, 'coef_'))
