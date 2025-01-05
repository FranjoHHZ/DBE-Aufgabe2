import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from TheAlgorithm import TheAlgorithm
from main import download_data,split

# Laden die Daten von Testdatafile ein
# Definieren des Verzeichnispfades
data_dir = 'data'

# Laden der Konfusionsmatrizen aus dem 'data' Verzeichnis
train_confusion_matrix = np.load(os.path.join(data_dir, 'train_confusion_matrix.npy'))
test_confusion_matrix = np.load(os.path.join(data_dir, 'test_confusion_matrix.npy'))

# Laden der Genauigkeiten aus der Textdatei im 'data' Verzeichnis
with open(os.path.join(data_dir, 'accuracy.txt'), 'r') as f:
    accuracies = f.readlines()
    train_accuracy = accuracies[0].strip()
    test_accuracy = accuracies[1].strip()

class TestInput(unittest.TestCase):
    # Laden die Daten von Testdatafile ein
    # Definieren des Verzeichnispfades
    data_dir = 'data'

    # Laden der Konfusionsmatrizen aus dem 'data' Verzeichnis
    train_confusion_matrix = np.load(os.path.join(data_dir, 'train_confusion_matrix.npy'))
    test_confusion_matrix = np.load(os.path.join(data_dir, 'test_confusion_matrix.npy'))

    # Laden der Genauigkeiten aus der Textdatei im 'data' Verzeichnis
    with open(os.path.join(data_dir, 'accuracy.txt'), 'r') as f:
        accuracies = f.readlines()
        train_accuracy = float(accuracies[0].strip())
        test_accuracy = float(accuracies[1].strip())

  
    @classmethod
    def setUpClass(cls):
        # print('setupClass')   
        pass

    @classmethod
    def tearDownClass(cls): 
        # print('teardownClass')
        pass

    def setUp(self):
        print('setUp') 
        X, y = download_data()
        splitRatio = 60000
        self.X_train, self.y_train, self.X_test, self.y_test = split(X,y,splitRatio) 
        self.train_accuracy = train_accuracy
        self.train_confusion_matrix = train_confusion_matrix
        
        self.test_accuracy = test_accuracy
        self.test_confusion_matrix = test_confusion_matrix

    def tearDown(self):
        # print('tearDown')
        pass
        
    def test_fit(self):     
        np.random.seed(31337)
        self.ta = TheAlgorithm(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertEqual(self.ta.fit(), float(self.train_accuracy)) 
        self.assertEqual(self.ta.train_confusion_matrix.tolist(), self.train_confusion_matrix.tolist())  
  
    def test_predict(self):
        np.random.seed(31337)
        self.ta = TheAlgorithm(self.X_train, self.y_train, self.X_test, self.y_test)
        self.ta.fit()
        self.assertEqual(self.ta.predict(), float(self.test_accuracy))
        self.assertEqual(self.ta.train_confusion_matrix.tolist(), self.train_confusion_matrix.tolist()) 
      
if __name__ == '__main__':
  
    #run tests 
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
