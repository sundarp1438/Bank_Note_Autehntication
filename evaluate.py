import os
import yaml
import argparse
import psutil
import json
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(config_path):
    """Evaluates the trained ANN model, logs system metrics, and saves performance metrics with MLflow tracking."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    test_path = config['data_split']['testset_path']
    model_path = config['train']['model_path']
    metrics_path = config['evaluate']['metrics_path']
    system_metrics_path = config['evaluate']['system_metrics_path']
    
    # Load test dataset
    df = pd.read_csv(test_path)
    target_column = config['featurize']['target_column']
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]
    
    # Load trained model
    model = tf.keras.models.load_model(model_path)
    
    # Start MLflow tracking
    mlflow.set_experiment("BankNote_ANN_Evaluation")
    with mlflow.start_run():
        mlflow.tensorflow.log_model(model, "evaluated_model")  # Log model
        
        # Make predictions
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log accuracy
        mlflow.log_metric("accuracy", accuracy)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        
        # Log classification report metrics
        for label, metrics in report.items():
            if isinstance(metrics, dict):  # Ensure it's a dictionary before logging
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", metric_value)
        
        # Ensure metrics directory exists
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        os.makedirs(os.path.dirname(system_metrics_path), exist_ok=True)
        
        # Save metrics to JSON file
        metrics = {
            "accuracy": accuracy,
            "classification_report": report
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Log metrics file as an artifact in MLflow
        mlflow.log_artifact(metrics_path)
        
        # Collect system metrics
        system_metrics = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent
        }
        
        # Save system metrics to JSON file
        with open(system_metrics_path, 'w') as f:
            json.dump(system_metrics, f, indent=4)
        
        # Log system metrics to MLflow
        for key, value in system_metrics.items():
            mlflow.log_metric(f"system_{key}", value)
        
        mlflow.log_artifact(system_metrics_path)
        
        print(f"✅ Model evaluation complete. Metrics saved at: {metrics_path}")
        print(f"✅ System metrics saved at: {system_metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to config file")
    args = parser.parse_args()
    
    evaluate_model(args.config)
