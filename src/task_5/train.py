import os
import datetime
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from mlflow.models.signature import infer_signature

from src.task_5.train_pipeline import get_preprocessing_pipeline
from src.task_5.data_split import split_data
from src.task_5.evaluation import compute_metrics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MLRUNS_PATH = os.path.join(PROJECT_ROOT, "mlruns")

mlflow.set_tracking_uri(f"file:{MLRUNS_PATH}")
mlflow.set_experiment("credit-risk-pipeline-demo")


def train_and_log_pipeline(X_train, X_test, y_train, y_test, save_folder=None):
    if save_folder is None:
        save_folder = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(save_folder, exist_ok=True)

    pipeline = Pipeline([
        ("preprocessor", get_preprocessing_pipeline()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    raw_predictions = pipeline.predict_proba(X_train)[:, 1]
    signature = infer_signature(X_train, raw_predictions)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="pipeline_model",
        registered_model_name="credit_risk_pipeline",
        signature=signature,
        input_example=X_train.head(1)
    )

    # Save locally
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_path = os.path.join(
        save_folder, f"credit_risk_pipeline_{timestamp}.pkl")
    joblib.dump(pipeline, pipeline_path)
    print(f"✅ Pipeline saved locally at: {pipeline_path}")

    return metrics


def main():
    df = pd.read_csv("data/processed/clean_data.csv")
    TARGET = "is_high_risk"

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    unique_targets = df[TARGET].dropna().unique()
    if len(unique_targets) != 2:
        raise ValueError("Target must be binary!")

    sorted_targets = sorted(unique_targets)
    df[TARGET] = df[TARGET].map({sorted_targets[0]: 0, sorted_targets[1]: 1})
    print(f"✅ '{TARGET}' remapped to binary: {df[TARGET].unique()}")

    X_train, X_test, y_train, y_test = split_data(df, target=TARGET)

    metrics = train_and_log_pipeline(X_train, X_test, y_train, y_test)
    print("✅ Training complete. Metrics:", metrics)


if __name__ == "__main__":
    main()
