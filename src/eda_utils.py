"""SBA EDA에서 재사용하는 순수 함수."""
import numpy as np
import pandas as pd


def parse_money(s: pd.Series) -> pd.Series:
    """'$60,000.00 ' 같은 통화 문자열을 float로 변환. 변환 불가/빈값은 NaN."""
    cleaned = (
        s.astype("string")
        .str.replace(r'[\$",\s]', "", regex=True)
        .replace("", pd.NA)
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def binarize_target(s: pd.Series) -> pd.Series:
    """MIS_Status -> CHGOFF=1, 'P I F'=0, 그 외/결측=NaN (float Series)."""
    mapping = {"CHGOFF": 1.0, "P I F": 0.0}
    return s.map(mapping).astype("float64")
