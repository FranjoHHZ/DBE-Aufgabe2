from sklearn.linear_model import LogisticRegression
from src.decorators import my_logger, my_timer
from src.Unit_download import Normalize

class TheAlgorithm:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    @my_logger
    @my_timer
    def fit(self):
        normalizer = Normalize()
        self.X_train, self.X_test = normalizer.normalize(self.X_train, self.X_test)

        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

    @my_logger
    @my_timer
    def predict(self):
        return self.model.predict(self.X_test)
