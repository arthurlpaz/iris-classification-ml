from sklearn.model_selection import train_test_split, cross_val_score
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.models.train_model import train_model
from src.models.evaluate import evaluate_model
from src.models.save_model import save_model
import yaml


def run_pipeline():
    configs = yaml.safe_load(open("configs/config.yaml"))

    df = load_data()
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=configs["data"]["test_size"],
        random_state=configs["data"]["random_state"],
    )

    model = train_model(X_train, y_train, configs)

    cv_scores = cross_val_score(model, X, y, cv=5)

    print(f"CV Accuracy Mean: {cv_scores.mean():.4f}")
    print(f"CV Accuracy Std: {cv_scores.std():.4f}")
    accuracy = evaluate_model(model, X_test, y_test)

    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Cross-Validation Scores: {cv_scores}")

    save_model(model)
