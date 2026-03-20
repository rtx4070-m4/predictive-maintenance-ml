
from preprocessing import generate_synthetic_data, scale_features, create_sequences
from model_lstm import build_model
from tensorflow.keras.callbacks import EarlyStopping
import os

def train():
    df = generate_synthetic_data()
    df, scaler = scale_features(df)
    X, y = create_sequences(df)

    model = build_model((X.shape[1], X.shape[2]))
    es = EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(X, y, epochs=10, validation_split=0.2, callbacks=[es])

    os.makedirs("../models", exist_ok=True)
    model.save("../models/lstm_model.h5")

if __name__ == "__main__":
    train()
