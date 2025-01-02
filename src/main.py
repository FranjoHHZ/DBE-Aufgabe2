from Unit_download import download, split
from TheAlgorithm import TheAlgorithm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    # Daten laden
    X, y = download()
    X_train, y_train, X_test, y_test = split(X, y, split_ratio=60000)

    # Modell trainieren und testen
    model = TheAlgorithm(X_train, y_train, X_test, y_test)
    model.fit()
    predictions = model.predict()

    # Ergebnisse berechnen
    train_accuracy = accuracy_score(y_train, model.model.predict(X_train))
    test_accuracy = accuracy_score(y_test, predictions)
    train_conf_matrix = confusion_matrix(y_train, model.model.predict(X_train))
    test_conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    # Ergebnisse in Datei speichern
    with open("ausgabe.txt", "w") as f:
        f.write(f"Train Accuracy: {train_accuracy * 100:.2f}%\n")
        f.write(f"Train confusion matrix:\n{train_conf_matrix}\n\n")
        f.write(f"Classification report for classifier:\n{class_report}\n")
        f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
        f.write(f"Test confusion matrix:\n{test_conf_matrix}\n")

if __name__ == "__main__":
    main()
