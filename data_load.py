import os
import pandas as pd
import yaml
import argparse
import mlflow


def load_data(config_path):
    """Loads dataset from CSV and saves it to the raw data folder, with MLflow tracking."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    dataset_path = config['data_load']['dataset_path']
    raw_data_dir = os.path.dirname(dataset_path)
    
    # Start MLflow run to track data loading
    with mlflow.start_run():
        # Log parameters related to data loading
        mlflow.log_param("dataset_url", "https://raw.githubusercontent.com/sundarp1438/Bank_Note_Autehntication/main/BankNoteAuthentication.csv")
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # Ensure raw data directory exists
        os.makedirs(raw_data_dir, exist_ok=True)
        
        # Load dataset
        df = pd.read_csv("https://raw.githubusercontent.com/sundarp1438/Bank_Note_Autehntication/main/BankNoteAuthentication.csv")
        
        # Save dataset locally
        df.to_csv(dataset_path, index=False)
        
        # Log the artifact (saved dataset)
        mlflow.log_artifact(dataset_path)
        
        print(f"✅ Data successfully loaded and saved at: {dataset_path}")
        
        # Log the success message in MLflow
        mlflow.log_metric("data_load_success", 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to config file")
    args = parser.parse_args()
    
    load_data(args.config)
