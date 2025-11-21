from statistics import median
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd


def build_preprocessing_pipeline(X: pd.DataFrame):
    numeric_features=list(X.columns)

    numeric_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy=median)),
        ("scaler",StandardScaler())
    ])

    preprocessor=ColumnTransformer([
        ("num",numeric_pipeline,numeric_features)
    ])

    return preprocessor


