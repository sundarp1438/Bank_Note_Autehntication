import os
import yaml
import json
import mlflow
import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric

def detect_data_drift(config_path):
    """Detects data drift between train and test datasets for ANN model and logs results to MLflow."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    train_path = config['data_split']['trainset_path']
    test_path = config['data_split']['testset_path']
    drift_report_path = config['evaluate']['drift_report_path']

    # Load train and test datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Remove target column to analyze only input features
    target_column = config['featurize']['target_column']
    X_train = df_train.drop(columns=[target_column])
    X_test = df_test.drop(columns=[target_column])

    # Create Evidently Data Drift Report
    drift_report = Report(metrics=[DataDriftTable(), DatasetDriftMetric()])
    drift_report.run(reference_data=X_train, current_data=X_test)

    # Save drift report as HTML
    os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)
    drift_report.save_html(drift_report_path)
    print(f"✅ Data drift report saved at {drift_report_path}")

    # Extract drift metrics
    drift_results = drift_report.as_dict()

    # Log drift metrics to MLflow
    mlflow.set_experiment("BankNote ANN Data Drift Detection")
    with mlflow.start_run():
        mlflow.log_dict(drift_results, "data_drift.json")
        mlflow.log_artifact(drift_report_path)
        print("✅ Data drift metrics and report logged to MLflow")

if __name__ == "__main__":
    detect_data_drift("params.yaml")
