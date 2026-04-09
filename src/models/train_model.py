from sklearn.ensemble import RandomForestClassifier
from src.configs import configs

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators = configs["model"]["n_estimators"],
        max_depth = configs["model"]["max_depth"],
        random_state = configs["model"]["random_state"]
    
    )
    model.fit(X_train, y_train)

    return model
