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


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix


def build_structured_transformer() -> ColumnTransformer:
    """수치=중앙값대치+표준화, 범주=최빈대치+원핫(min_frequency로 희소범주 묶음)."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=50))])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)])


def build_name_vectorizer() -> TfidfVectorizer:
    """사업체명 char n-gram TF-IDF (설계: char_wb (3,5), min_df=20, max_features=5000)."""
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           min_df=20, max_features=5000)


def transform_features(struct_transformer, name_vec, df, include_name: bool):
    """학습된 변환기로 정형(+선택적 Name) 희소행렬 생성."""
    Xs = csr_matrix(struct_transformer.transform(df))
    if not include_name:
        return Xs
    Xn = name_vec.transform(df["Name"].astype("string").fillna(""))
    return hstack([Xs, Xn]).tocsr()
