# tests/test_data.py
import pandas as pd

from src.data.load_data import load_data


def test_load_data_returns_dataframe():
    df = load_data()

    assert isinstance(df, pd.DataFrame)
    assert "target" in df.columns
    assert df.shape[0] == 150
    assert df["target"].nunique() == 3


def test_load_data_has_correct_columns():
    df = load_data()
    expected_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']
    
    assert list(df.columns) == expected_columns


def test_load_data_target_values():
    df = load_data()
    
    assert df["target"].min() == 0
    assert df["target"].max() == 2