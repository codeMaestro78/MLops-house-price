

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import numpy as np
from typing import List
import os



class PredictRequest(BaseModel):
    data:List[List[float]]

class PredictResponse(BaseModel):
    predictions:List[float]

app=FastAPI(title="House Price Predictor")

mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI","https://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

model=None

@app.on_event("startup")

async def load_model():
    global model

    try:
        client=mlflow.tracking.MlflowClient()
        experiments=client.list_experiments()
        if not experiments:
            print("NO experiments found")
            return
        runs=client.search_runs(experiment_ids=[experiments[0].experiments_id],
                                order_by=["start_time DESC"],max_results=1)
        if runs:
            run_id=runs[0].info.run_id
            uri=f"runs/{run_id}/model"
            model=mlflow.sklearn.load_model(uri)
            print(f"Loaded model from run {run_id}")
        else:
            print("no runs found in mlflow api will not predict until a model is trained and logged")

    except Exception as e:
        print("Failed to load model on startup ",e)


@app.get("/health")
async def health():
    return {"status":"ok","model_loaded":model is not None}


@app.post("/predict" , response_model=PredictResponse)
async def predict(req:PredictRequest):
    if model is None:
        return {"predictions":[]}
    arr=np.array(req.data)
    preds=model.predict(arr).tolist()
    return {"predctions":preds}
