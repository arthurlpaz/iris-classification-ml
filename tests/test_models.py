# tests/test_models.py
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.models.train_model import train_model
from src.models.evaluate import evaluate_model
from src.models.save_model import save_model


def test_train_model_returns_fitted_classifier():
    df = load_data()
    X, y = preprocess_data(df)
    configs = {"model": {"n_estimators": 5, "max_depth": 3}, "data": {"random_state": 42}}

    model = train_model(X, y, configs)
    predictions = model.predict(X.iloc[:5])

    assert len(predictions) == 5
    assert set(predictions).issubset({0, 1, 2})


def test_train_model_with_different_configs():
    df = load_data()
    X, y = preprocess_data(df)
    configs = {"model": {"n_estimators": 10, "max_depth": 5}, "data": {"random_state": 42}}

    model = train_model(X, y, configs)
    
    assert model.n_estimators == 10
    assert model.max_depth == 5


def test_evaluate_model_outputs_accuracy_in_range():
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train, {"model": {"n_estimators": 5, "max_depth": 3}, "data": {"random_state": 42}})
    accuracy = evaluate_model(model, X_test, y_test)

    assert 0.0 <= accuracy <= 1.0



