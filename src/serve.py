

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import numpy as np
from typing import List
import os
import pandas as pd

class PredictRequest(BaseModel):
    data: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[float]

app = FastAPI(title="House Price Predictor")

mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

model = None
EXPERIMENT_NAME = "house-prices-local"

@app.on_event("startup")
async def load_model():
    global model
    try:
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        # Try to load registered model first
        try:
            model = mlflow.sklearn.load_model("models:/house-prices-local/None")
            print("Loaded registered model 'house-prices-local'")
            return
        except Exception as e:
            print(f"Failed to load registered model: {e}")

        # Fallback to latest run in experiment
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments(filter_string=f"name = '{EXPERIMENT_NAME}'")
        if not experiments:
            print(f"No experiment named '{EXPERIMENT_NAME}' found")
            return
        experiment = experiments[0]
        runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                                  order_by=["start_time DESC"], max_results=1)
        if runs:
            run_id = runs[0].info.run_id
            uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(uri)
            print(f"Loaded model from run {run_id} in experiment '{EXPERIMENT_NAME}'")
            print(f"Model type: {type(model)}")
            if hasattr(model, 'named_steps'):
                print(f"Pipeline steps: {list(model.named_steps.keys())}")
        else:
            print(f"No runs found in experiment '{EXPERIMENT_NAME}'")
    except Exception as e:
        print(f"Failed to load model on startup: {e}")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    global model
    try:
        print("Incoming request data:", req.data)
        if model is None:
            print("Model not loaded")
            return {"predictions": []}

        arr = np.array(req.data)
        print("Input array shape:", arr.shape)

        # Convert to DataFrame with correct column names
        columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        df = pd.DataFrame(arr, columns=columns)
        print("Input DataFrame shape:", df.shape)
        print("Input DataFrame columns:", df.columns.tolist())

        preds = model.predict(df).tolist()
        print("Predictions:", preds)
        return {"predictions": preds}

    except Exception as e:
        import traceback
        print(f"Error in /predict: {e}")
        traceback.print_exc()
        return {"predictions": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
