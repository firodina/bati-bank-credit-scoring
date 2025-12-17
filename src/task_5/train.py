# ========================================================
# credit_risk_mlflow_training.py
# ========================================================
import pandas as pd
import os
import datetime
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.task_5.model_factory import get_models
from src.task_5.data_split import split_data
from src.task_5.evaluation import compute_metrics


# Set MLflow tracking location (local folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MLRUNS_PATH = os.path.join(PROJECT_ROOT, "mlruns")

mlflow.set_tracking_uri(f"file:{MLRUNS_PATH}")
mlflow.set_experiment("credit-risk-demo")


def train_and_log_model(
    model_name,
    model_config,
    X_train,
    X_test,
    y_train,
    y_test,
    save_folder=None
):
    if save_folder is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        save_folder = os.path.join(BASE_DIR, "models")

    os.makedirs(save_folder, exist_ok=True)
    """
    Train a model with hyperparameter tuning, log to MLflow, and save locally.
    """
    os.makedirs(save_folder, exist_ok=True)

    with mlflow.start_run(run_name=model_name):
        # Choose search type
        if model_config["search"] == "grid":
            search = GridSearchCV(
                model_config["model"],
                model_config["params"],
                scoring="roc_auc",
                cv=3,
                n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                model_config["model"],
                model_config["params"],
                scoring="roc_auc",
                cv=3,
                n_iter=10,
                random_state=42,
                n_jobs=-1
            )

        # Train model
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Predictions & metrics
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # Log params and metrics to MLflow
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            best_model, artifact_path="model", registered_model_name="credit_risk_model")

        # Save locally with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(save_folder, f"{model_name}_{timestamp}.pkl")
        joblib.dump(best_model, model_path)
        print(f"✅ Model saved locally at: {model_path}")

        return metrics


def main():

    df = pd.read_csv("data/processed/clean_data.csv")

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target="is_high_risk")

    # Get all models
    models = get_models()

    all_results = {}

    # Train each model separately
    for model_name, model_config in models.items():
        print(f"\n🔹 Training {model_name} ...")
        metrics = train_and_log_model(
            model_name,
            model_config,
            X_train,
            X_test,
            y_train,
            y_test
        )
        print(f"Metrics for {model_name}: {metrics}")
        all_results[model_name] = metrics

    return all_results


if __name__ == "__main__":
    results = main()
    print("\n===== ALL MODEL RESULTS =====")
    for model, metric in results.items():
        print(f"{model}: {metric}")
