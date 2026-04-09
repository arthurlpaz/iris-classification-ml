import joblib
import os

def save_model(model, path="models/model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)