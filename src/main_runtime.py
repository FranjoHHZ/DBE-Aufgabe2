import time
from Unit_download import download, split
from TheAlgorithm import TheAlgorithm

def measure_runtime():
    X, y = download()
    X_train, y_train, X_test, y_test = split(X, y, split_ratio=60000)
    model = TheAlgorithm(X_train, y_train, X_test, y_test)

    # Laufzeit messen
    start_time = time.time()
    model.fit()
    end_time = time.time()

    runtime = end_time - start_time

    # Laufzeit im Hauptverzeichnis speichern
    with open("baseline_runtime.txt", "w") as f:
        f.write(f"{runtime}\n")
    
    # Laufzeit in der Konsole anzeigen
    print(f"Laufzeit erfolgreich gemessen: {runtime:.2f} Sekunden")

if __name__ == "__main__":
    measure_runtime()

