import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_data(as_frame=True):
    data=fetch_california_housing(as_frame=True)
    X=data.frame.drop(columns=["MedHoueseVal"]) if as_frame else data.data
    y=data.frame["MedHouseVal"] if as_frame else data.target
    return X,y

if __name__=="__main__":
    X,y=load_data()
    print(f"Shape of X is {X.shape} and y is {y.shape}")

