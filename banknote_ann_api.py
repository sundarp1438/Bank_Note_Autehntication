import uvicorn
import yaml
import os
import pandas as pd
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

# ✅ Load Config
with open("params.yaml", "r") as file:
    config = yaml.safe_load(file)

MODEL_PATH = config["train"]["model_path"]
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("BankNote ANN Prediction API")

# ✅ Load ANN Model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🚨 Model file not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Initialize FastAPI
app = FastAPI(title="BankNote ANN Prediction API", description="Predicts banknote authenticity using an ANN model.")

# ✅ Define Request Body
class BankNoteInput(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

@app.get("/")
def home():
    return {"message": "Welcome to the BankNote ANN Prediction API!"}

@app.post("/predict")
def predict(data: BankNoteInput):
    """Predicts banknote authenticity (class: 0 or 1) using the ANN model and logs results to MLflow."""
    input_data = pd.DataFrame([data.dict()])

    # Ensure input data matches ANN input shape
    model_input_shape = model.input_shape[1]  # ANN expects a specific number of features
    if input_data.shape[1] != model_input_shape:
        raise ValueError(f"🚨 Expected {model_input_shape} input features, but got {input_data.shape[1]}")

    # Make Prediction
    prediction = model.predict(input_data)[0][0]
    predicted_class = int(prediction > 0.5)  # Convert probability to binary class (0 or 1)

    # ✅ Log input & output to MLflow
    with mlflow.start_run():
        mlflow.log_params(data.dict())
        mlflow.log_metric("predicted_class", predicted_class)

    return {
        "predicted_class": predicted_class,
        "message": "Banknote authenticity predicted successfully and logged to MLflow."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
