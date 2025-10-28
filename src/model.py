import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def build_model_pipeline(preprocessor):
    model=RandomForestRegressor(n_estimators=200,random_state=42)
    pipe =Pipeline([
        ("preprocessor",preprocessor),
        ("model",model)
    ])

    return pipe


def log_model_to_mlflow(pipeline,run_name:str,model_name:str=None):
    mlflow.sklearn.log_model(pipeline,artifact_path="model")
    if model_name:
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri,model_name)



