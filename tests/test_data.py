import numpy as np
import pandas as pd
from src.data import filter_modeling, time_split

def _toy():
    return pd.DataFrame({
        "ApprovalFY": [2005, 2006, 2007, 2009, 2012, 2006],
        "target":     [0.0,  1.0,  0.0,  1.0,  0.0,  np.nan],
        "NewExist":   ["1",  "2",  "1",  "2",  "1",  "1"],
    })

def test_filter_modeling_drops_nan_target_future_and_bad_newexist():
    df = _toy().copy()
    df.loc[6] = [2005, 0.0, "0"]
    out = filter_modeling(df)
    assert out["target"].notna().all()
    assert (out["ApprovalFY"] <= 2010).all()
    assert set(out["NewExist"].unique()) <= {"1", "2"}
    assert len(out) == 4

def test_time_split_boundaries():
    df = filter_modeling(_toy())
    parts = time_split(df)
    assert set(parts["train"]["ApprovalFY"]) <= {2005, 2006}
    assert set(parts["valid"]["ApprovalFY"]) == {2007}
    assert set(parts["test"]["ApprovalFY"]) <= {2008, 2009, 2010}
    assert len(parts["train"]) + len(parts["valid"]) + len(parts["test"]) == len(df)
