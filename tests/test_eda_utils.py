import numpy as np
import pandas as pd
from src.eda_utils import parse_money, binarize_target

def test_parse_money_handles_dollar_comma_space():
    s = pd.Series(['$60,000.00 ', '$0.00 ', '"$1,234.50"', None, ''])
    out = parse_money(s)
    assert out.tolist()[:3] == [60000.0, 0.0, 1234.5]
    assert np.isnan(out.iloc[3]) and np.isnan(out.iloc[4])

def test_binarize_target_maps_chgoff_pif_and_drops_unknown():
    s = pd.Series(['CHGOFF', 'P I F', None, 'WTF'])
    out = binarize_target(s)
    assert out.tolist()[0] == 1
    assert out.tolist()[1] == 0
    assert np.isnan(out.iloc[2]) and np.isnan(out.iloc[3])
