import argparse
from os import name
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from src.data import load_data
from src.preprocess import build_preprocessing_pipeline
from src.model import build_model_pipeline


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--experiment-name",default="house-prices-demo")
    p.add_argument("--test-size",type=float,default=0.2)
    p.add_argument("--random-state",type=int,default=42)
    p.add_argument("--register",action="store_true",help="Register model in MLflow Model Registry")
    return p.parse_args()

def main():
    args=parse_args()
    X,y=load_data()

    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=args.test_size,random_state=args.random_state)
    preprocessor= build_preprocessing_pipeline(X_train)
    pipeline=build_model_pipeline(preprocessor)

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run() as run:
        pipeline.fit(X_train,y_train)
        y_pred=pipeline.predict(X_val)
        rmse=root_mean_squared_error(y_val,y_pred)
        mlflow.log_metric("rmse",rmse)
        mlflow.log_param("n_estimators",200)


        # log and optinally register
        mlflow.sklearn.log_model(pipeline,artifact_path="model")
        if args.register:
            model_uri=f"runs:/{run.info.run_id}/model"

            try:
                mlflow.register_model(model_uri,"house_price_model")
            except Exception as e:
                print("model registry may not be available local ",e)

        print(f"Finished run {run.info.run_id} RMSE = {rmse:.4f}")


if __name__=="__main__":
    main()


#  important
# prefer running python -m src.train from project root to avoid imports issues, or ensure PYTHONPATH=.
