from sklearn.datasets import load_iris
import pandas as pd

def load_data():
    data = load_iris()
    df = data.frame
    df['target'] = data.target
    return df
