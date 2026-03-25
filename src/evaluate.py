import pandas as pd
import joblib
import numpy as np
from scipy.stats import entropy

def evaluate():
    # Load data
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    y_test = y_test.values.ravel()

    # Load model
    model = joblib.load("models/model.pkl")

    probs = model.predict_proba(X_test)

    conf_threshold = 0.8
    entropy_threshold = 0.5

    correct = 0
    total = len(y_test)
    abstained = 0

    for i in range(total):
        confidence = np.max(probs[i])
        ent = entropy(probs[i])

        # Abstain condition
        if confidence < conf_threshold or ent > entropy_threshold:
            abstained += 1
            continue

        prediction = np.argmax(probs[i])

        if prediction == y_test[i]:
            correct += 1

    valid_preds = total - abstained
    accuracy = correct / valid_preds if valid_preds > 0 else 0

    print("Final Accuracy:", accuracy)
    print("Abstained:", abstained)
    print("Total:", total)

if __name__ == "__main__":
    evaluate()