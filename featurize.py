import os
import pandas as pd
import yaml
import argparse
import mlflow
from sklearn.preprocessing import StandardScaler

def featurize_data(config_path):
    """Preprocesses the dataset by scaling numerical features, with MLflow tracking."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    raw_data_path = config['data_load']['dataset_path']
    processed_data_path = config['featurize']['processed_path']
    target_column = config['featurize']['target_column']
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("raw_data_path", raw_data_path)
        mlflow.log_param("processed_data_path", processed_data_path)
        mlflow.log_param("target_column", target_column)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # Load dataset
        df = pd.read_csv(raw_data_path)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert back to DataFrame
        processed_df = pd.DataFrame(X_scaled, columns=X.columns)
        processed_df[target_column] = y.values
        
        # Ensure processed data directory exists
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        
        # Save processed dataset
        processed_df.to_csv(processed_data_path, index=False)
        print(f"✅ Processed data saved at: {processed_data_path}")

        # Log the artifact (processed data file)
        mlflow.log_artifact(processed_data_path)

        # Log metrics (basic statistics on feature scaling)
        mlflow.log_metric("num_features", X.shape[1])
        mlflow.log_metric("num_samples", X.shape[0])
        mlflow.log_metric("scaling_mean", scaler.mean_.mean())
        mlflow.log_metric("scaling_var", scaler.var_.mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to config file")
    args = parser.parse_args()
    
    featurize_data(args.config)
