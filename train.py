import os
import yaml
import argparse
import pandas as pd
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def train_model(config_path):
    """Trains an Artificial Neural Network (ANN) on the dataset with MLflow tracking."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    train_path = config['data_split']['trainset_path']
    model_path = config['train']['model_path']
    input_shape = config['train']['input_shape']
    hidden_layers = config['train']['hidden_layers']
    activation = config['train']['activation']
    output_activation = config['train']['output_activation']
    loss = config['train']['loss']
    optimizer = config['train']['optimizer']
    metrics = config['train']['metrics']
    epochs = config['train']['epochs']
    batch_size = config['train']['batch_size']
    
    # Load training dataset
    df = pd.read_csv(train_path)
    target_column = config['featurize']['target_column']
    X_train = df.drop(columns=[target_column])
    y_train = df[target_column]
    
    # Start MLflow run
    mlflow.tensorflow.autolog()  # Enables automatic logging of TensorFlow models

    with mlflow.start_run():
        # Log training parameters
        mlflow.log_param("train_path", train_path)
        mlflow.log_param("input_shape", input_shape)
        mlflow.log_param("hidden_layers", hidden_layers)
        mlflow.log_param("activation", activation)
        mlflow.log_param("output_activation", output_activation)
        mlflow.log_param("loss", loss)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # Define ANN model
        model = Sequential()
        model.add(Dense(hidden_layers[0], activation=activation, input_shape=(input_shape,)))
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation))
        model.add(Dense(1, activation=output_activation))
        
        # Compile model
        model.compile(loss=loss, optimizer=Adam(), metrics=metrics)
        
        # Train model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        # Log final training loss & metrics
        final_loss = history.history['loss'][-1]
        mlflow.log_metric("final_loss", final_loss)

        # Ensure model directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save trained model
        model.save(model_path)
        print(f"✅ Model saved at: {model_path}")

        # Log model artifact in MLflow
        mlflow.tensorflow.log_model(model, "trained_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train_model(args.config)
