from src.preprocess import build_preprocessing_pipeline
import pandas as pd
import numpy as np

def test_preprocess_builds_pipeline():
    X=pd.DataFrame({
        'a':[1.0,2.0,None],
        'b':[0.5,None,0.1]
    })

    pre=build_preprocessing_pipeline(X)
    out=pre.fit_transform(X)
    assert out.shape[1]==2
    assert not any(np.isnan(out.flatten()))
