import unittest
from ..src.Unit_download import download, split
from ..src.TheAlgorithm import TheAlgorithm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class TestPredict(unittest.TestCase):
    def setUp(self):
        X, y = download()
        self.X_train, self.y_train, self.X_test, self.y_test = split(X, y, split_ratio=60000)
        self.model = TheAlgorithm(self.X_train, self.y_train, self.X_test, self.y_test)
        self.model.fit()

    def test_predict(self):
        predictions = self.model.predict()
        accuracy = accuracy_score(self.y_test, predictions)
        conf_matrix = confusion_matrix(self.y_test, predictions)
        class_report = classification_report(self.y_test, predictions)

        # Ergebnisse speichern
        with open("ausgabe2.txt", "w") as f:
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
            f.write(f"Confusion matrix:\n{conf_matrix}\n")
            f.write(f"Classification report:\n{class_report}\n")
