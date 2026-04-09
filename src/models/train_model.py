from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, config):
    model = RandomForestClassifier(
        n_estimators = config["model"]["n_estimators"],
        max_depth = config["model"]["max_depth"],
        random_state = config["data"]["random_state"]
    
    )
    model.fit(X_train, y_train)

    return model
