import joblib

def save_model(model, path="models/iris_model.pkl"):
    joblib.dump(model, path)