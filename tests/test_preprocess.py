# tests/test_preprocess.py
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data


def test_preprocess_data_separates_features_and_target():
    df = load_data()
    X, y = preprocess_data(df)

    assert "target" not in X.columns
    assert len(X) == len(y) == df.shape[0]
    assert y.name == "target"


def test_preprocess_data_X_shape():
    df = load_data()
    X, y = preprocess_data(df)
    
    assert X.shape[1] == 4  # 4 features
    assert X.shape[0] == 150


def test_preprocess_data_y_values():
    df = load_data()
    X, y = preprocess_data(df)
    
    assert set(y.unique()) == {0, 1, 2}