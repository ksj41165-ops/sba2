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
