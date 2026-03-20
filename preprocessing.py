
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=n_samples, freq="H"),
        "vibration": np.random.normal(0.5, 0.1, n_samples),
        "temperature": np.random.normal(70, 5, n_samples),
        "power_draw": np.random.normal(200, 20, n_samples),
        "equipment_id": np.random.randint(1, 5, n_samples)
    })
    data["failure_label"] = ((data["vibration"] > 0.7) | (data["temperature"] > 80)).astype(int)
    return data

def scale_features(df):
    scaler = MinMaxScaler()
    features = ["vibration", "temperature", "power_draw"]
    df[features] = scaler.fit_transform(df[features])
    return df, scaler

def create_sequences(df, seq_length=20):
    features = ["vibration", "temperature", "power_draw"]
    X, y = [], []
    for i in range(len(df) - seq_length):
        X.append(df[features].iloc[i:i+seq_length].values)
        y.append(df["failure_label"].iloc[i+seq_length])
    return np.array(X), np.array(y)
