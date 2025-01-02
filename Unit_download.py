from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

def download():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.astype('float64')
    y = mnist.target
    return (X, y)

class Normalize:
    def normalize(self, X_train, X_test):
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return (X_train, X_test)

    def inverse(self, X):
        return self.scaler.inverse_transform(X)

def split(X, y, split_ratio):
    X_train = X[:split_ratio]
    y_train = y[:split_ratio]
    X_test = X[split_ratio:]
    y_test = y[split_ratio:]
    return (X_train, y_train, X_test, y_test)
