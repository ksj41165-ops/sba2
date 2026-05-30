# SBA 부실예측 모델링 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SBA 대출 데이터로 승인시점 부실예측 모델(로지스틱 베이스라인 + LightGBM 주력)을 학습하고, "정형만 vs 정형+사업체명 TF-IDF" 비교 및 thin-file proxy 세그먼트 성능까지 산출한다.

**Architecture:** 순수 함수/변환기를 `src/`에 두고(단위테스트 가능), 무거운 학습·평가는 노트북 `notebooks/02_sba_modeling.ipynb`에서 오케스트레이션한다. 모든 전처리(스케일러/원핫/TF-IDF)는 **train에만 fit**하고 valid/test는 transform만 한다. 정형 피처는 ColumnTransformer로 희소행렬화하고 Name TF-IDF와 hstack하여 두 모델이 동일한 피처표현(sparse)을 공유한다 — 이로써 ablation이 깔끔해진다.

**Tech Stack:** Python 3.13, pandas 2.3, numpy, scipy 1.16(sparse), scikit-learn 1.7(ColumnTransformer/OneHotEncoder/TfidfVectorizer/LogisticRegression/metrics), LightGBM 4.6, matplotlib.

설계 문서: `docs/superpowers/specs/2026-05-30-sba-default-prediction-guide-design.md`

---

## File Structure

- Create: `src/data.py` — 로딩/필터/시간분할 (load_clean, filter_modeling, time_split)
- Create: `src/features.py` — 파생피처(add_features) + 전처리 변환기(build_structured_transformer, build_name_vectorizer, transform_features) + 피처 상수
- Create: `src/modeling.py` — 평가지표(evaluate) + 세그먼트 평가(segment_metrics)
- Create: `tests/test_data.py`, `tests/test_features.py`, `tests/test_modeling.py`
- Create: `notebooks/02_sba_modeling.ipynb` — 학습·평가·비교표·세그먼트·그림
- Create: `reports/` outputs (modeling_results.csv, pr_curves.png) — .gitignore의 `reports/figures/`는 그림용이므로 CSV는 `reports/`에 두되 커밋 허용

데이터 경로: 리포 루트 `SBAnational.csv`. 기존 `src/eda_utils.py`(parse_money, binarize_target) 재사용. 모든 테스트는 `python -m pytest`로 리포 루트에서 실행(소형 합성 DataFrame 사용, CSV 미사용 → 빠름).

피처 정의(승인시점, 누수 제외):
- 수치: `GrAppv, SBA_Appv, sba_ratio(=SBA_Appv/GrAppv), Term, NoEmp, CreateJob, RetainedJob`
- 범주: `NewExist, is_franchise, UrbanRural, RevLineCr, LowDoc, State, BankState, naics2(NAICS 앞2자리)`
- `ApprovalFY`는 split 키로만 쓰고 피처에서 제외(시간 누수/미관측 범주 회피). `Name`은 TF-IDF 전용.

---

## Task 1: `src/data.py` — 로딩/필터/시간분할 (TDD)

**Files:**
- Create: `src/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data.py
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
    df.loc[6] = [2005, 0.0, "0"]  # NewExist outlier -> dropped
    out = filter_modeling(df)
    assert out["target"].notna().all()
    assert (out["ApprovalFY"] <= 2010).all()
    assert set(out["NewExist"].unique()) <= {"1", "2"}
    assert len(out) == 4  # rows 0,1,2,3 ; row4 future, row5 nan target, row6 bad newexist

def test_time_split_boundaries():
    df = filter_modeling(_toy())
    parts = time_split(df)
    assert set(parts["train"]["ApprovalFY"]) <= {2005, 2006}
    assert set(parts["valid"]["ApprovalFY"]) == {2007}
    assert set(parts["test"]["ApprovalFY"]) <= {2008, 2009, 2010}
    assert len(parts["train"]) + len(parts["valid"]) + len(parts["test"]) == len(df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/data.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git -c user.name='Claude' -c user.email='noreply@anthropic.com' commit -m "feat: add data load/filter/time_split with tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `src/features.py` — 파생피처 add_features (TDD)

**Files:**
- Create: `src/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
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
    assert np.isnan(out["sba_ratio"].iloc[2])           # GrAppv==0 -> inf -> NaN
    assert out["is_franchise"].tolist() == ["0", "1", "0"]  # 0/NaN -> "0", 5 -> "1"
    assert out["naics2"].tolist() == ["45", "72", "0"]
    assert out["is_new"].tolist() == [0, 1, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py -v`
Expected: FAIL — `ImportError: cannot import name 'add_features'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/features.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_features.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git -c user.name='Claude' -c user.email='noreply@anthropic.com' commit -m "feat: add_features (sba_ratio, is_franchise, naics2, is_new) with tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `src/features.py` — 전처리 변환기 + 결합 (TDD)

**Files:**
- Modify: `src/features.py` (append functions)
- Test: `tests/test_features.py` (append tests)

- [ ] **Step 1: Write the failing test (append)**

```python
# append to tests/test_features.py
from scipy.sparse import issparse
from src.features import (build_structured_transformer, build_name_vectorizer,
                          transform_features)

def _toy_full(n=40):
    import numpy as np
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
    assert Xsn.shape[1] > Xs.shape[1]   # Name 추가로 열 증가
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_structured_transformer'`

- [ ] **Step 3: Write minimal implementation (append to src/features.py)**

```python
# append to src/features.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_features.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git -c user.name='Claude' -c user.email='noreply@anthropic.com' commit -m "feat: structured transformer + name vectorizer + transform_features with tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `src/modeling.py` — 평가지표 (TDD)

**Files:**
- Create: `src/modeling.py`
- Test: `tests/test_modeling.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_modeling.py
import numpy as np
from src.modeling import evaluate, segment_metrics

def test_evaluate_perfect_scores():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate(y, s, threshold=0.5)
    assert set(m) == {"pr_auc", "roc_auc", "precision", "recall", "f1"}
    assert abs(m["roc_auc"] - 1.0) < 1e-9
    assert abs(m["recall"] - 1.0) < 1e-9
    assert abs(m["precision"] - 1.0) < 1e-9

def test_segment_metrics_keys_and_subsetting():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.2, 0.7, 0.4, 0.6])
    is_new = np.array([1, 1, 0, 0])
    out = segment_metrics(y, s, is_new, threshold=0.5)
    assert set(out) == {"overall", "new(thin-file)", "existing"}
    # overall uses all 4; new uses first 2; existing uses last 2
    assert out["new(thin-file)"]["recall"] == 1.0   # y=1 at idx1, s=0.7>=0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modeling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.modeling'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/modeling.py
"""부실예측 평가지표 (불균형 대응: PR-AUC 핵심)."""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)


def evaluate(y_true, y_score, threshold: float = 0.5) -> dict:
    """확률점수 기반 PR-AUC/ROC-AUC + 임계값 기반 precision/recall/f1."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def segment_metrics(y_true, y_score, is_new, threshold: float = 0.5) -> dict:
    """overall / new(thin-file proxy) / existing 세그먼트별 evaluate."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    is_new = np.asarray(is_new)
    segs = {
        "overall": np.ones(len(y_true), dtype=bool),
        "new(thin-file)": is_new == 1,
        "existing": is_new == 0,
    }
    return {k: evaluate(y_true[m], y_score[m], threshold) for k, m in segs.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modeling.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/modeling.py tests/test_modeling.py
git -c user.name='Claude' -c user.email='noreply@anthropic.com' commit -m "feat: evaluate + segment_metrics with tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: 모델링 노트북 — 로딩/분할/전처리 + 로지스틱 (정형, 정형+Name)

**Files:**
- Create: `notebooks/02_sba_modeling.ipynb`

전체 노트북은 매 태스크 `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/02_sba_modeling.ipynb` 로 실행(대용량+학습으로 시간 소요). 한글 출력 UTF-8 보존을 위해 env 필수. 셀은 끝에 append, 이전 셀 수정 금지. NEVER git-add SBAnational.csv.

- [ ] **Step 1: CELL 1 (code) — 임포트/로딩/필터/분할/전처리 적합**

```python
import sys
from pathlib import Path
ROOT = Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src.data import load_clean, filter_modeling, time_split
from src.features import (add_features, build_structured_transformer,
                          build_name_vectorizer, transform_features)
from src.modeling import evaluate, segment_metrics

REP = ROOT / "reports"; (REP / "figures").mkdir(parents=True, exist_ok=True)

df = add_features(filter_modeling(load_clean(ROOT / "SBAnational.csv")))
parts = time_split(df)
for k, v in parts.items():
    print(k, v.shape, "부실률:", round(v["target"].mean(), 4))

st = build_structured_transformer().fit(parts["train"])
nv = build_name_vectorizer().fit(parts["train"]["Name"].astype("string").fillna(""))
```

- [ ] **Step 2: CELL 2 (code) — 피처행렬 생성 (정형 / 정형+Name)**

```python
def make_xy(part, include_name):
    X = transform_features(st, nv, part, include_name=include_name)
    y = part["target"].astype(int).values
    return X, y

Xtr_s, ytr = make_xy(parts["train"], False)
Xte_s, yte = make_xy(parts["test"], False)
Xtr_sn, _ = make_xy(parts["train"], True)
Xte_sn, _ = make_xy(parts["test"], True)
is_new_test = parts["test"]["is_new"].values
print("정형:", Xtr_s.shape, "| 정형+Name:", Xtr_sn.shape)
```

- [ ] **Step 3: CELL 3 (code) — 로지스틱 회귀 (balanced) 학습·평가**

```python
from sklearn.linear_model import LogisticRegression

results = {}
def fit_eval_logreg(Xtr, Xte, tag):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    results[tag] = evaluate(yte, s)
    print(tag, results[tag])
    return clf, s

logreg_s, s_logreg_s = fit_eval_logreg(Xtr_s, Xte_s, "logreg | 정형")
logreg_sn, s_logreg_sn = fit_eval_logreg(Xtr_sn, Xte_sn, "logreg | 정형+Name")
```

- [ ] **Step 4: Execute & verify**

Run: `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/02_sba_modeling.ipynb`
Expected: 에러 없이 완료. train/valid/test shape와 부실률 출력(train 부실률은 2006 이전이므로 test 2008-2010보다 낮게 나옴). logreg 두 줄의 pr_auc/roc_auc 출력. 정형+Name의 pr_auc가 정형 단독보다 같거나 높게 나오는지 관찰(보장은 아님).

- [ ] **Step 5: Commit**

```bash
git add notebooks/02_sba_modeling.ipynb
git -c user.name='Claude' -c user.email='noreply@anthropic.com' commit -m "feat: modeling notebook with logistic baseline (structured, +name)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: 모델링 노트북 — LightGBM (정형, 정형+Name)

**Files:**
- Modify: `notebooks/02_sba_modeling.ipynb` (append)

- [ ] **Step 1: CELL 4 (code) — scale_pos_weight 계산 + LightGBM 학습·평가**

```python
import lightgbm as lgb

spw = (ytr == 0).sum() / (ytr == 1).sum()
print("scale_pos_weight:", round(spw, 3))

def fit_eval_lgbm(Xtr, Xte, tag):
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=spw, n_jobs=-1, random_state=0)
    clf.fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    results[tag] = evaluate(yte, s)
    print(tag, results[tag])
    return clf, s

lgbm_s, s_lgbm_s = fit_eval_lgbm(Xtr_s, Xte_s, "lgbm | 정형")
lgbm_sn, s_lgbm_sn = fit_eval_lgbm(Xtr_sn, Xte_sn, "lgbm | 정형+Name")
```

- [ ] **Step 2: Execute & verify**

Run: `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/02_sba_modeling.ipynb`
Expected: 에러 없이 완료. scale_pos_weight 출력(대략 3~5 사이; train 부실률 낮음 반영). lgbm 두 줄 pr_auc/roc_auc 출력. `results` 딕셔너리에 4개 키 존재.

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_sba_modeling.ipynb
git -c user.name='Claude' -c user.email='noreply@anthropic.com' commit -m "feat: LightGBM models (structured, +name) in modeling notebook

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: 모델링 노트북 — 비교표 + thin-file 세그먼트 + 그림/CSV 저장

**Files:**
- Modify: `notebooks/02_sba_modeling.ipynb` (append)

- [ ] **Step 1: CELL 5 (code) — 4개 모델 비교표 저장**

```python
res_df = pd.DataFrame(results).T[["pr_auc", "roc_auc", "precision", "recall", "f1"]].round(4)
res_df.index.name = "model"
display(res_df)
res_df.to_csv(REP / "modeling_results.csv", encoding="utf-8-sig")
print("saved:", REP / "modeling_results.csv")
```

- [ ] **Step 2: CELL 6 (code) — 최고 모델의 thin-file 세그먼트 성능**

```python
best_tag = res_df["pr_auc"].idxmax()
best_scores = {"logreg | 정형": s_logreg_s, "logreg | 정형+Name": s_logreg_sn,
               "lgbm | 정형": s_lgbm_s, "lgbm | 정형+Name": s_lgbm_sn}[best_tag]
print("최고 PR-AUC 모델:", best_tag)
seg = segment_metrics(yte, best_scores, is_new_test)
seg_df = pd.DataFrame(seg).T.round(4)
display(seg_df)
seg_df.to_csv(REP / "segment_results.csv", encoding="utf-8-sig")
```

- [ ] **Step 3: CELL 7 (code) — PR 곡선 그림 저장**

```python
from sklearn.metrics import precision_recall_curve
score_map = {"logreg | 정형": s_logreg_s, "logreg | 정형+Name": s_logreg_sn,
             "lgbm | 정형": s_lgbm_s, "lgbm | 정형+Name": s_lgbm_sn}
fig, ax = plt.subplots(figsize=(6, 5))
for tag, s in score_map.items():
    p, r, _ = precision_recall_curve(yte, s)
    ax.plot(r, p, label=f"{tag} (AP={results[tag]['pr_auc']:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curves (test 2008-2010)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(REP / "figures" / "pr_curves.png", dpi=120); plt.show()
```

- [ ] **Step 4: CELL 8 (markdown) — 결과 요약**

```markdown
## 모델링 결과 요약
- 평가: test = 2008~2010 승인 코호트 (금융위기 영향으로 보수적)
- 핵심 지표 PR-AUC 기준 4개 모델(로지스틱/LightGBM × 정형/정형+Name) 비교표 참조
- 비정형(Name TF-IDF) 추가 효과: 비교표의 "정형" vs "정형+Name" PR-AUC 차이로 정량화
- thin-file(신규사업자) proxy 세그먼트 성능은 segment_results.csv 참조
- 산출물: reports/modeling_results.csv, reports/segment_results.csv, reports/figures/pr_curves.png
```

- [ ] **Step 5: Execute final & verify**

Run: `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/02_sba_modeling.ipynb`
Expected: 전체 무오류. `reports/modeling_results.csv`(4행), `reports/segment_results.csv`(3행), `reports/figures/pr_curves.png` 생성. 비교표/세그먼트표 출력 표시. 한글 마크다운 정상.

- [ ] **Step 6: Commit**

```bash
git add notebooks/02_sba_modeling.ipynb reports/modeling_results.csv reports/segment_results.csv
git -c user.name='Claude' -c user.email='noreply@anthropic.com' commit -m "feat: model comparison table, thin-file segment, PR curves

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

(주의: `reports/figures/`는 .gitignore로 제외되어 pr_curves.png는 커밋되지 않음 — 의도된 동작. CSV는 reports/ 직하라 커밋됨.)

---

## Self-Review

**Spec coverage:**
- 예측시점 승인(누수 제외, ApprovalFY 피처제외) → Task 1 filter + Task 2/3 피처셋 ✅
- 타깃 CHGOFF=1 (eda_utils.binarize_target) → Task 1 load_clean ✅
- thin-file proxy(NewExist=2) 세그먼트 리포팅 → Task 2 is_new + Task 4 segment_metrics + Task 7 ✅
- 정형 피처(금액/조건/범주, sba_ratio, naics2) → Task 2/3 ✅
- 비정형 Name TF-IDF(char_wb 3-5, min_df 20, max_features 5000) → Task 3 ✅
- 정형 vs 정형+Name 비교 → Task 5/6/7 (include_name 토글) ✅
- 고카디널리티 인코딩(OneHot min_frequency; naics2 2자리) → Task 2/3 ✅
- 로지스틱 베이스라인 + LightGBM 주력 → Task 5/6 ✅
- 불균형(class_weight=balanced, scale_pos_weight) → Task 5/6 ✅
- time split train<=2006/valid2007/test2008-2010, 2011+ 제외 → Task 1 ✅
- 평가지표 PR-AUC 핵심 + ROC/Precision/Recall/F1 → Task 4/7 ✅

**Placeholder scan:** Task 7 Step 3에 의도적으로 "잘못된 셀 → 정확한 코드 사용" 안내가 있음 — 구현자는 정확한 코드 블록만 넣을 것. 그 외 TBD/TODO 없음.

**Type consistency:** `transform_features(st, nv, df, include_name=)` 시그니처 Task 3 정의와 Task 5 사용 일치. `evaluate`/`segment_metrics` Task 4 정의와 Task 5/6/7 사용 일치. `results` 딕셔너리 키("logreg | 정형" 등)와 `score_map` 키 동일. `is_new`(Task2)·`is_new_test`(Task5)·segment_metrics(Task4) 정합.

**참고(valid 미사용):** 본 계획은 고정 하이퍼파라미터를 사용하므로 valid셋은 분할만 하고 튜닝엔 쓰지 않는다(YAGNI). 향후 튜닝 시 valid 활용.
