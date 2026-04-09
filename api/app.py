from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/model.pkl")

FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=10)
    sepal_width: float = Field(..., gt=0, lt=10)
    petal_length: float = Field(..., gt=0, lt=10)
    petal_width: float = Field(..., gt=0, lt=10)


@app.post("/predict")
def predict(data: IrisInput):
    df = pd.DataFrame([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]], columns=FEATURE_NAMES)

    pred = model.predict(df)[0]
    probs = model.predict_proba(df)[0]

    return {
        "prediction": int(pred),
        "class": CLASS_NAMES[pred],
        "confidence": float(max(probs))
    }