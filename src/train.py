import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train():
    # Load data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    # Convert y to 1D
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Start MLflow
    mlflow.set_experiment("heart-disease")

    with mlflow.start_run():
        # Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)

        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    train()