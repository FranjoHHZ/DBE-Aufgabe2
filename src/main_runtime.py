import numpy as np
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from TheAlgorithm import TheAlgorithm
from main import download_data, split

def save_baseline_runtime():
    # Daten vorbereiten
    X, y = download_data()
    splitRatio = 60000
    X_train, y_train, X_test, y_test = split(X, y, splitRatio)

    # Instanz von TheAlgorithm erstellen
    ta = TheAlgorithm(X_train, y_train, X_test, y_test)

    # Startzeit messen
    start_time = time.time()

    # fit()-Methode ausführen
    ta.fit()

    # Endzeit messen und Laufzeit berechnen
    end_time = time.time()
    runtime = end_time - start_time

    # Laufzeit in einer Datei speichern
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    baseline_file_path = os.path.join(data_dir, 'baseline_runtime.txt')
    
    # Prüfen, ob die Datei existiert und ob sie leer ist
    separator = ';'  # Trennzeichen festlegen
    need_separator = os.path.exists(baseline_file_path) and os.path.getsize(baseline_file_path) > 0
    
    with open(baseline_file_path, 'a') as file:  # 'a' für Anhängen
        if need_separator:
            file.write(separator)  # Trennzeichen hinzufügen, falls nötig
        file.write(f"{runtime}")  # Neue Laufzeit hinzufügen
    
    print(f"Die Laufzeit des fit()-Aufrufs wurde an {baseline_file_path} angehängt: {runtime} Sekunden")

if __name__ == '__main__':
    save_baseline_runtime()

