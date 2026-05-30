"""SBA 데이터 로딩·필터·시간분할."""
import pandas as pd
from src.eda_utils import parse_money, binarize_target

MONEY_COLS = ["DisbursementGross", "BalanceGross", "ChgOffPrinGr", "GrAppv", "SBA_Appv"]
NUM_COLS = ["Term", "NoEmp", "CreateJob", "RetainedJob", "ApprovalFY", "FranchiseCode", "UrbanRural"]


def load_clean(path) -> pd.DataFrame:
    """원본 CSV를 읽어 금액/수치/타깃을 파싱한 DataFrame 반환."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    for c in MONEY_COLS:
        df[c] = parse_money(df[c])
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["target"] = binarize_target(df["MIS_Status"])
    return df


def filter_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """타깃 결측 제외, 절단연도(ApprovalFY>=2011) 제외, NewExist 이상치 제외."""
    m = df["target"].notna() & (df["ApprovalFY"] <= 2010) & df["NewExist"].isin(["1", "2"])
    return df.loc[m].copy()


def time_split(df: pd.DataFrame) -> dict:
    """ApprovalFY 기준 train<=2006 / valid==2007 / test 2008-2010."""
    fy = df["ApprovalFY"]
    return {
        "train": df[fy <= 2006].copy(),
        "valid": df[fy == 2007].copy(),
        "test": df[(fy >= 2008) & (fy <= 2010)].copy(),
    }
