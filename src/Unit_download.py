from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

def download():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.astype('float64')
    y = mnist.target
    return (X, y)

class Normalize(object): 
    def normalize(self, X_train, X_test):
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)
        return (X_train, X_test) 
    
    def inverse(self, X_train, X_val, X_test):
        X_train = self.scaler.inverse_transform(X_train)
        X_test  = self.scaler.inverse_transform(X_test)
        return (X_train, X_test)   

def split(X,y, splitRatio):
    X_train = X[:splitRatio]
    y_train = y[:splitRatio]
    X_test = X[splitRatio:]
    y_test = y[splitRatio:]
    return (X_train, y_train, X_test, y_test)  
