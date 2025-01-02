import time
from src.Unit_download import download, split
from src.TheAlgorithm import TheAlgorithm

def measure_runtime():
    X, y = download()
    X_train, y_train, X_test, y_test = split(X, y, split_ratio=60000)
    model = TheAlgorithm(X_train, y_train, X_test, y_test)

    start_time = time.time()
    model.fit()
    end_time = time.time()

    runtime = end_time - start_time
    with open("data/baseline_runtime.txt", "w") as f:
        f.write(f"{runtime}\n")
    print(f"Baseline runtime: {runtime} seconds")

if __name__ == "__main__":
    measure_runtime()
