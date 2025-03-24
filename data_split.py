import os
import pandas as pd
import yaml
import argparse
import mlflow
from sklearn.model_selection import train_test_split

def split_data(config_path):
    """Splits the dataset into training and testing sets, with MLflow tracking."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    processed_data_path = config['featurize']['processed_path']
    train_path = config['data_split']['trainset_path']
    test_path = config['data_split']['testset_path']
    test_size = config['data_split']['test_size']
    random_state = config['base']['random_state']
    
    # Start MLflow tracking
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("processed_data_path", processed_data_path)
        mlflow.log_param("train_path", train_path)
        mlflow.log_param("test_path", test_path)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # Load processed dataset
        df = pd.read_csv(processed_data_path)
        
        # Split dataset into train and test sets
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        
        # Save split datasets
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Log artifacts (train and test datasets)
        mlflow.log_artifact(train_path)
        mlflow.log_artifact(test_path)

        # Log metrics
        mlflow.log_metric("train_size", len(train_df))
        mlflow.log_metric("test_size", len(test_df))

        print(f"✅ Train set saved at: {train_path}")
        print(f"✅ Test set saved at: {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to config file")
    args = parser.parse_args()
    
    split_data(args.config)
