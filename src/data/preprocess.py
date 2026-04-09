def preprocess_data(df):
    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y
