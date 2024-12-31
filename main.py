from functools import wraps
import logging
import time

def my_logger(orig_func):
    logging.basicConfig(filename=f'{orig_func.__name__}.log', level=logging.INFO)
    
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(f'Ran with args: {args}, and kwargs: {kwargs}')
        return orig_func(*args, **kwargs)
    return wrapper

def my_timer(orig_func):
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = orig_func(*args, **kwargs)
        end_time = time.time()
        logging.info(f'{orig_func.__name__} ran in: {end_time - start_time:.4f} sec')
        return result
    return wrapper

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

class TheAlgorithm:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    @my_logger
    @my_timer
    def fit(self):
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

    @my_logger
    @my_timer
    def predict(self, X):
        return self.model.predict(X
