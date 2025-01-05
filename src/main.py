# Importieren notwendiger Bibliotheken
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from decorators import my_logger, my_timer 
from TheAlgorithm import TheAlgorithm
import os

# Eventuell erforderliche zusätzliche Importe
# from normalizer import Normalize

# Funktion zum Herunterladen und Vorbereiten der MNIST-Daten
def download_data():
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.astype('float64')
    y = mnist.target
    return X, y

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

#The solution
if __name__ == '__main__': 
  
  X,y = download_data()
  print ('MNIST:', X.shape, y.shape)
  
  splitRatio = 60000
  X_train, y_train, X_test, y_test = split(X,y,splitRatio) 

  np.random.seed(31337)
  ta = TheAlgorithm(X_train, y_train, X_test, y_test)
  train_accuracy = ta.fit()
  print()
  print('Train Accuracy:', train_accuracy,'\n') 
  print("Train confusion matrix:\n%s\n" % ta.train_confusion_matrix)
  
  test_accuracy = ta.predict()
  print()
  print('Test Accuracy:', test_accuracy,'\n') 
  print("Test confusion matrix:\n%s\n" % ta.test_confusion_matrix)

  # Speichern das Ergebnis in einer Datei im Unterverzeichnis 'data'
  data_dir = 'data'
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  # Speichern der Konfusionsmatrizen im 'data' Verzeichnis
  np.save(os.path.join(data_dir, 'train_confusion_matrix.npy'), ta.train_confusion_matrix)
  np.save(os.path.join(data_dir, 'test_confusion_matrix.npy'), ta.test_confusion_matrix)

  # Speichern der Genauigkeiten in eine Textdatei im 'data' Verzeichnis
  with open(os.path.join(data_dir, 'accuracy.txt'), 'w') as f:
     f.write(f'{train_accuracy}\n')
     f.write(f'{test_accuracy}\n')
