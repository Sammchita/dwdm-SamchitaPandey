from sklearn import metrics
from sklearn.svm import SVC

# FIX 1: Removed unused import — `datasets` was imported but never used


def main() -> None:
    food = {
        "data": [
            [220, 12, 8, 5],
            [180, 9, 5, 4],
            [320, 14, 18, 6],
            [150, 7, 4, 3],
            [410, 20, 22, 8],
            [95, 4, 1, 2],
            [130, 6, 3, 2],
            [360, 16, 19, 7],
            [280, 11, 10, 5],
            [70, 3, 1, 1],
            # FIX 2: Removed stray lines "20" and "Roll No- 79010263"
            #         that were breaking the list literal
            [240, 10, 9, 4],
            [390, 18, 21, 7],
        ],
        "target": [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        "feature_names": ["Calories", "Protein", "Fat", "Sugar"],
        "target_names": ["Unhealthy", "Healthy"],
    }

    x = food["data"]
    y = food["target"]

    print("Length of Data:", len(x))

    split = int(len(x) * 0.7)
    trainx, testx = x[:split], x[split:]
    trainy, testy = y[:split], y[split:]

    print("Number of features:", len(food["feature_names"]))
    print("Number of classes: ", len(food["target_names"]))
    print("Class Labels:      ", food["target_names"])

    model = SVC(kernel="linear")
    model.fit(trainx, trainy)
    yp = model.predict(testx)

    print("\nConfusion Matrix:")
    print(metrics.confusion_matrix(testy, yp))

    print("\nClassification Measures:")
    print("Accuracy: ", metrics.accuracy_score(testy, yp))
    print("Recall:   ", metrics.recall_score(testy, yp, zero_division=0))   # FIX 3: added zero_division=0
    print("Precision:", metrics.precision_score(testy, yp, zero_division=0)) # FIX 3: added zero_division=0
    print("F1-score: ", metrics.f1_score(testy, yp, zero_division=0))       # FIX 3: added zero_division=0


if __name__ == "__main__":   # FIX 4: added indentation to the guard block
    main()