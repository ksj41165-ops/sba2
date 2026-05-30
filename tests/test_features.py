import numpy as np
import pandas as pd
from src.features import add_features

def _toy():
    return pd.DataFrame({
        "GrAppv":       [100000.0, 50000.0, 0.0],
        "SBA_Appv":     [80000.0, 25000.0, 10000.0],
        "FranchiseCode":[0.0, 5.0, np.nan],
        "NAICS":        ["451120", "722410", "0"],
        "NewExist":     ["1", "2", "1"],
        "UrbanRural":   [1.0, 2.0, 0.0],
    })

def test_add_features_creates_expected_columns():
    out = add_features(_toy())
    assert abs(out["sba_ratio"].iloc[0] - 0.8) < 1e-9
    assert np.isnan(out["sba_ratio"].iloc[2])
    assert out["is_franchise"].tolist() == ["0", "1", "0"]
    assert out["naics2"].tolist() == ["45", "72", "0"]
    assert out["is_new"].tolist() == [0, 1, 0]

from scipy.sparse import issparse
from src.features import (build_structured_transformer, build_name_vectorizer,
                          transform_features)

def _toy_full(n=120):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "GrAppv": rng.uniform(1e4, 5e5, n), "SBA_Appv": rng.uniform(1e3, 4e5, n),
        "Term": rng.integers(0, 200, n), "NoEmp": rng.integers(0, 50, n),
        "CreateJob": rng.integers(0, 10, n), "RetainedJob": rng.integers(0, 10, n),
        "FranchiseCode": rng.integers(0, 6, n).astype(float),
        "NAICS": rng.choice(["451120", "722410", "236220"], n),
        "NewExist": rng.choice(["1", "2"], n), "UrbanRural": rng.integers(0, 3, n).astype(float),
        "RevLineCr": rng.choice(["Y", "N"], n), "LowDoc": rng.choice(["Y", "N"], n),
        "State": rng.choice(["IN", "OH", "CA"], n), "BankState": rng.choice(["IN", "OH"], n),
        "Name": rng.choice(["ABC MOTEL", "XYZ TRUCKING LLC", "JOE NAIL SALON"], n),
    })
    return add_features(df)

def test_transformers_fit_on_train_transform_shapes_match():
    df = _toy_full()
    st = build_structured_transformer().fit(df)
    nv = build_name_vectorizer().fit(df["Name"].astype("string").fillna(""))
    Xs = transform_features(st, nv, df, include_name=False)
    Xsn = transform_features(st, nv, df, include_name=True)
    assert issparse(Xs) and issparse(Xsn)
    assert Xs.shape[0] == len(df) == Xsn.shape[0]
    assert Xsn.shape[1] > Xs.shape[1]
