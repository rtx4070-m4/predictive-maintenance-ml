
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("../models/lstm_model.h5")

def predict(sequence):
    sequence = np.array(sequence).reshape(1, len(sequence), len(sequence[0]))
    prob = model.predict(sequence)[0][0]
    return float(prob)
