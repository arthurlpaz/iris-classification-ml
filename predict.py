import joblib
import pandas as pd

FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def load_model(path="models/model.pkl"):
    return joblib.load(path)


def predict(features):
    model = load_model()

    df = pd.DataFrame([features], columns=FEATURE_NAMES)

    pred = model.predict(df)[0]
    probs = model.predict_proba(df)[0]


    return {
        "prediction": int(pred),
        "class": CLASS_NAMES[pred],
        "confidence": float(max(probs))
    }


if __name__ == "__main__":
    sample = [1.1, 5.5, 5.4, 5.2]
    result = predict(sample)
    print(result)