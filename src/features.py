"""승인시점 피처 생성 및 전처리 변환기."""
import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["GrAppv", "SBA_Appv", "sba_ratio", "Term", "NoEmp", "CreateJob", "RetainedJob"]
CATEGORICAL_FEATURES = ["NewExist", "is_franchise", "UrbanRural", "RevLineCr", "LowDoc",
                        "State", "BankState", "naics2"]


def _franchise_flag(x) -> str:
    return "1" if pd.notna(x) and x not in (0, 1) else "0"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """sba_ratio, is_franchise, naics2, is_new 및 범주형 문자열화를 추가한 복사본."""
    df = df.copy()
    ratio = df["SBA_Appv"] / df["GrAppv"]
    df["sba_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)
    df["is_franchise"] = df["FranchiseCode"].map(_franchise_flag)
    df["naics2"] = df["NAICS"].astype("string").str[:2]
    df["UrbanRural"] = df["UrbanRural"].astype("string")
    df["is_new"] = df["NewExist"].map({"1": 0, "2": 1})
    return df
