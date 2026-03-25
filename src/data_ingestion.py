import pandas as pd
import os

def load_data():
    data_path = "../data/raw/heart.csv"
    df = pd.read_csv(data_path)
    return df

def save_data(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/processed.csv", index=False)

if __name__ == "__main__":
    df = load_data()
    save_data(df)
    print("Data ingestion completed!")