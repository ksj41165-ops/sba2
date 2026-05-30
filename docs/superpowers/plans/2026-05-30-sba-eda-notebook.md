# SBA 부실예측 EDA 노트북 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SBA 대출 데이터(`SBAnational.csv`, 89만 행)에 대해 설계 문서의 EDA 항목 9단계를 수행하는 Jupyter 노트북을 만들어, 모델링 단계에서 쓸 피처 후보·split 경계·누수 위험을 확정한다.

**Architecture:** 단일 노트북 `notebooks/01_sba_eda.ipynb`를 섹션별 셀로 구성한다. 재사용 유틸(금액 파싱)은 `src/eda_utils.py`로 분리해 노트북과 향후 모델링이 공유한다. 노트북은 `jupyter nbconvert --execute`로 통째 실행해 에러 없이 끝나는지(=재현성)와 알려진 기준값(행수 899,164 / 부실률 ~17.6%)이 재현되는지로 검증한다.

**Tech Stack:** Python 3.13, pandas 2.3, numpy, scikit-learn 1.7(TfidfVectorizer), matplotlib, seaborn, Jupyter(nbconvert).

설계 문서: `docs/superpowers/specs/2026-05-30-sba-default-prediction-guide-design.md`

---

## File Structure

- Create: `src/eda_utils.py` — 금액 문자열 파싱, 타깃 이진화 등 순수 함수(테스트 가능)
- Create: `tests/test_eda_utils.py` — `eda_utils` 단위 테스트
- Create: `notebooks/01_sba_eda.ipynb` — EDA 본체(섹션별 셀)
- Create: `reports/figures/` — 노트북이 저장하는 그림(.gitignore에 추가)
- Modify: `.gitignore` — `reports/figures/`, `*.ipynb` 체크포인트 등

데이터 경로는 리포 루트 기준 `SBAnational.csv`. 노트북은 리포 루트에서 실행한다고 가정.

---

## Task 1: 재사용 유틸 `eda_utils` (TDD)

**Files:**
- Create: `src/eda_utils.py`
- Test: `tests/test_eda_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eda_utils.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eda_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eda_utils'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/eda_utils.py
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
    return pd.to_numeric(cleaned, errors="coerce")


def binarize_target(s: pd.Series) -> pd.Series:
    """MIS_Status -> CHGOFF=1, 'P I F'=0, 그 외/결측=NaN (float Series)."""
    mapping = {"CHGOFF": 1.0, "P I F": 0.0}
    return s.map(mapping).astype("float64")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eda_utils.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/eda_utils.py tests/test_eda_utils.py
git commit -m "feat: add eda_utils (parse_money, binarize_target) with tests"
```

---

## Task 2: 노트북 스캐폴드 + 데이터 로딩/타입 정리 (설계 EDA 1단계)

**Files:**
- Create: `notebooks/01_sba_eda.ipynb`
- Modify: `.gitignore`

- [ ] **Step 1: `.gitignore`에 산출물 경로 추가**

`.gitignore`에 다음 줄을 추가(이미 있으면 생략):

```
reports/figures/
.ipynb_checkpoints/
```

- [ ] **Step 2: 노트북 생성 — 셀 1: 임포트/경로/그림 디렉터리**

`notebooks/01_sba_eda.ipynb`를 만들고 첫 코드 셀에 작성:

```python
import sys, os
from pathlib import Path
ROOT = Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.eda_utils import parse_money, binarize_target

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CSV = ROOT / "SBAnational.csv"
pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid")
print("csv exists:", CSV.exists())
```

- [ ] **Step 3: 셀 2 — 로딩 + 타입 정리**

```python
MONEY_COLS = ["DisbursementGross", "BalanceGross", "ChgOffPrinGr", "GrAppv", "SBA_Appv"]
df = pd.read_csv(CSV, dtype=str, low_memory=False)
for c in MONEY_COLS:
    df[c] = parse_money(df[c])
for c in ["Term", "NoEmp", "CreateJob", "RetainedJob", "ApprovalFY", "FranchiseCode", "UrbanRural"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["ApprovalDate"] = pd.to_datetime(df["ApprovalDate"], errors="coerce", format="%d-%b-%y")
df["target"] = binarize_target(df["MIS_Status"])
print("shape:", df.shape)
df.head(3)
```

- [ ] **Step 4: 노트북 실행해 에러 없는지 확인**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_sba_eda.ipynb`
Expected: 에러 없이 완료. 노트북을 열면 `shape: (899164, 29)` 출력(원본 27열 + target). `csv exists: True`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/01_sba_eda.ipynb .gitignore
git commit -m "feat: EDA notebook scaffold with loading and type parsing"
```

---

## Task 3: NewExist 의미 검증 (설계 EDA 2단계)

**Files:**
- Modify: `notebooks/01_sba_eda.ipynb`

- [ ] **Step 1: 셀 추가 — NewExist 분포 + 공식 정의 대조**

```python
# NewExist 공식 정의: 1 = Existing business, 2 = New business
print(df["NewExist"].value_counts(dropna=False))
# 이상치(0)와 결측은 분석에서 제외 표시
df["is_new"] = df["NewExist"].map({"1": 0, "2": 1})  # 1=신규(New)
print("\nis_new(1=신규) 분포:")
print(df["is_new"].value_counts(dropna=False))
```

- [ ] **Step 2: 셀 추가 — NewExist별 부실률로 정의 타당성 교차검증**

```python
chk = df.dropna(subset=["target"]).groupby("NewExist")["target"].agg(["mean", "size"])
print("NewExist별 부실률(target mean) / 건수:")
print(chk)
```

- [ ] **Step 3: 노트북 실행**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_sba_eda.ipynb`
Expected: NewExist 분포가 `1: 644869, 2: 253125, 0: 1034, NaN: 136`로 재현. 신규(2)와 기존(1)의 부실률이 출력됨(해석 방향 검증).

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_sba_eda.ipynb
git commit -m "feat: NewExist verification cells in EDA notebook"
```

---

## Task 4: 결측·이상치 점검 (설계 EDA 3단계)

**Files:**
- Modify: `notebooks/01_sba_eda.ipynb`

- [ ] **Step 1: 셀 추가 — 컬럼별 결측률**

```python
miss = df.isna().mean().sort_values(ascending=False)
print("결측률 상위:")
print((miss[miss > 0] * 100).round(2).astype(str) + "%")
```

- [ ] **Step 2: 셀 추가 — 수치 피처 이상치 요약**

```python
num_cols = ["GrAppv", "SBA_Appv", "Term", "NoEmp", "CreateJob", "RetainedJob"]
display(df[num_cols].describe(percentiles=[.01, .5, .99]).T)
# Term=0, GrAppv<=0 등 이상치 건수
print("Term==0:", (df["Term"] == 0).sum())
print("GrAppv<=0:", (df["GrAppv"] <= 0).sum())
print("NoEmp==0:", (df["NoEmp"] == 0).sum())
```

- [ ] **Step 3: 노트북 실행**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_sba_eda.ipynb`
Expected: 에러 없이 결측률 표와 describe 표, 이상치 카운트 출력.

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_sba_eda.ipynb
git commit -m "feat: missingness and outlier check cells"
```

---

## Task 5: 타깃 분포 및 불균형 (설계 EDA 4단계)

**Files:**
- Modify: `notebooks/01_sba_eda.ipynb`

- [ ] **Step 1: 셀 추가 — 타깃 분포 + 불균형 + 그림 저장**

```python
tgt = df["target"].value_counts(dropna=False)
print(tgt)
rate = df["target"].mean()
print(f"\n부실률(CHGOFF=1): {rate:.4f}")
print(f"타깃 결측(분석 제외 대상): {df['target'].isna().sum()}")

fig, ax = plt.subplots(figsize=(4, 3))
df["target"].dropna().map({0: "P I F", 1: "CHGOFF"}).value_counts().plot.bar(ax=ax)
ax.set_title("Target distribution"); ax.set_ylabel("count")
fig.tight_layout(); fig.savefig(FIG / "target_dist.png", dpi=120); plt.show()
```

- [ ] **Step 2: 노트북 실행**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_sba_eda.ipynb`
Expected: `부실률(CHGOFF=1): 0.17...` (약 0.176), 타깃 결측 1997, `reports/figures/target_dist.png` 생성.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_sba_eda.ipynb
git commit -m "feat: target distribution and imbalance cells"
```

---

## Task 6: 정형 피처별 부실률 관계 (설계 EDA 5단계)

**Files:**
- Modify: `notebooks/01_sba_eda.ipynb`

- [ ] **Step 1: 셀 추가 — NAICS 2자리 대분류 파생 + 업종별 부실률**

```python
d = df.dropna(subset=["target"]).copy()
d["naics2"] = df["NAICS"].str[:2]
by_naics = d.groupby("naics2")["target"].agg(["mean", "size"]).sort_values("mean", ascending=False)
by_naics = by_naics[by_naics["size"] >= 500]
display(by_naics.head(10))
```

- [ ] **Step 2: 셀 추가 — State / 승인액 구간 / Term 구간별 부실률**

```python
by_state = d.groupby("State")["target"].agg(["mean", "size"]).sort_values("mean", ascending=False)
display(by_state[by_state["size"] >= 500].head(10))

d["grappv_bin"] = pd.qcut(d["GrAppv"], 5, duplicates="drop")
d["term_bin"] = pd.cut(d["Term"], bins=[-1, 0, 60, 120, 180, 1000])
print("\n승인액 5분위별 부실률:"); print(d.groupby("grappv_bin", observed=True)["target"].mean())
print("\nTerm 구간별 부실률:"); print(d.groupby("term_bin", observed=True)["target"].mean())
```

- [ ] **Step 3: 셀 추가 — 부실률 막대그림 저장(업종 상위)**

```python
fig, ax = plt.subplots(figsize=(7, 4))
by_naics["mean"].head(10).plot.bar(ax=ax)
ax.set_title("Default rate by NAICS-2 (top 10)"); ax.set_ylabel("default rate")
fig.tight_layout(); fig.savefig(FIG / "default_by_naics2.png", dpi=120); plt.show()
```

- [ ] **Step 4: 노트북 실행**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_sba_eda.ipynb`
Expected: 업종/주/승인액/Term별 부실률 표 출력, `default_by_naics2.png` 생성. 에러 없음.

- [ ] **Step 5: Commit**

```bash
git add notebooks/01_sba_eda.ipynb
git commit -m "feat: structured feature vs default rate cells"
```

---

## Task 7: 신규사업자(thin-file proxy) 세그먼트 vs 전체 (설계 EDA 6단계)

**Files:**
- Modify: `notebooks/01_sba_eda.ipynb`

- [ ] **Step 1: 셀 추가 — 세그먼트 비교표**

```python
seg = d.copy()
seg["segment"] = seg["is_new"].map({1: "New(thin-file proxy)", 0: "Existing"})
cmp = seg.dropna(subset=["segment"]).groupby("segment").agg(
    n=("target", "size"),
    default_rate=("target", "mean"),
    avg_grappv=("GrAppv", "mean"),
    avg_term=("Term", "mean"),
)
print("주의: New는 신용이력이 아닌 '사업운영이력' 짧음 기반 proxy임")
display(cmp)
```

- [ ] **Step 2: 노트북 실행**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_sba_eda.ipynb`
Expected: New vs Existing 두 행의 건수/부실률/평균승인액/평균만기 비교표 출력.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_sba_eda.ipynb
git commit -m "feat: thin-file proxy segment vs overall comparison"
```

---

## Task 8: Name 텍스트 탐색 (설계 EDA 7단계)

**Files:**
- Modify: `notebooks/01_sba_eda.ipynb`

- [ ] **Step 1: 셀 추가 — 이름 길이/빈출 토큰**

```python
names = df["Name"].astype("string").fillna("")
print("이름 길이 describe:"); print(names.str.len().describe())

from collections import Counter
tokens = Counter()
for nm in names.sample(50000, random_state=0):
    tokens.update(nm.upper().split())
print("\n빈출 토큰 30:")
for w, c in tokens.most_common(30):
    print(f"{w:20s} {c}")
```

- [ ] **Step 2: 셀 추가 — TF-IDF char n-gram 설정 데모 + 미래암시 단어 점검**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=20, max_features=5000)
X_name = vec.fit_transform(names.fillna(""))
print("TF-IDF shape:", X_name.shape)

leak_words = ["BANKRUPT", "RECOVERY", "LIQUIDAT", "CLOSED", "FORECLOS"]
for w in leak_words:
    n = names.str.upper().str.contains(w, na=False).sum()
    print(f"미래암시 후보 '{w}': {n}건")
```

- [ ] **Step 3: 노트북 실행**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_sba_eda.ipynb`
Expected: 이름 길이 통계, 빈출 토큰, `TF-IDF shape: (899164, <=5000)`, 미래암시 단어 카운트 출력. 에러 없음.

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_sba_eda.ipynb
git commit -m "feat: Name text exploration and TF-IDF demo cells"
```

---

## Task 9: 누수 후보 컬럼 점검 (설계 EDA 8단계)

**Files:**
- Modify: `notebooks/01_sba_eda.ipynb`

- [ ] **Step 1: 셀 추가 — 승인 이후 관측 컬럼이 타깃과 결정적으로 연결되는지 확인**

```python
# 설계상 제외 대상: DisbursementDate, DisbursementGross, BalanceGross, ChgOffDate, ChgOffPrinGr
print("ChgOffDate 존재 & 부실 여부 교차:")
print(pd.crosstab(df["ChgOffDate"].notna(), df["target"]))
print("\nChgOffPrinGr>0 & 부실 여부 교차(누수 확인):")
print(pd.crosstab(df["ChgOffPrinGr"].fillna(0) > 0, df["target"]))
print("\n승인시점 모델에서 사용할 피처 화이트리스트:")
APPROVAL_FEATURES = ["GrAppv","SBA_Appv","Term","NoEmp","CreateJob","RetainedJob",
                     "NewExist","FranchiseCode","UrbanRural","RevLineCr","LowDoc",
                     "State","BankState","NAICS","ApprovalFY","Name"]
print(APPROVAL_FEATURES)
```

- [ ] **Step 2: 노트북 실행**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_sba_eda.ipynb`
Expected: `ChgOffDate` 존재 여부와 부실의 강한 연관(누수 근거) crosstab 출력, 화이트리스트 출력.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_sba_eda.ipynb
git commit -m "feat: leakage candidate check cells"
```

---

## Task 10: EDA 요약 & 모델링 시사점 + split 경계 제안 (설계 EDA 9단계)

**Files:**
- Modify: `notebooks/01_sba_eda.ipynb`

- [ ] **Step 1: 셀 추가 — ApprovalFY 분포로 train/valid/test 경계 확인**

```python
fy = pd.to_numeric(df["ApprovalFY"], errors="coerce")
print("ApprovalFY 분포(연도별 건수):")
print(fy.value_counts().sort_index())
print("\n제안 split: train<=2010 / valid 2011-2012 / test 2013-2014 (분포 보고 조정)")
```

- [ ] **Step 2: 마크다운 셀 추가 — EDA 요약**

마크다운 셀로 다음 내용 작성(실행 결과 수치를 채워 넣음):

```markdown
## EDA 요약 & 모델링 시사점
- 타깃 부실률: 약 17.6% (불균형 → PR-AUC 핵심 지표, scale_pos_weight)
- NewExist: 1=Existing, 2=New 확인. New 세그먼트를 thin-file proxy로 별도 리포팅
- 누수 컬럼(DisbursementGross/BalanceGross/ChgOff*/DisbursementDate) 제외 확정
- 승인시점 피처 화이트리스트(APPROVAL_FEATURES) 확정
- 고카디널리티: NAICS는 2자리 축소, Bank는 native categorical/target encoding
- Name TF-IDF(char 3-5, min_df, max_features=5000) 결합 실험 준비 완료
- split 경계: ApprovalFY 분포 기반 train/valid/test (위 셀 참조)
```

- [ ] **Step 3: 노트북 전체 재실행(처음부터 끝까지 재현성 최종 확인)**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_sba_eda.ipynb`
Expected: 전체 셀이 에러 없이 실행 완료(재현성 확인). `reports/figures/`에 그림 3개 생성.

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_sba_eda.ipynb
git commit -m "feat: EDA summary, modeling implications, split proposal"
```

---

## Self-Review

**Spec coverage (설계 EDA 1~9단계):**
1. 로딩·타입정리 → Task 2 ✅
2. NewExist 검증 → Task 3 ✅
3. 결측·이상치 → Task 4 ✅
4. 타깃 분포/불균형 → Task 5 ✅
5. 정형 피처별 부실률 → Task 6 ✅
6. 신규사업자 세그먼트 비교 → Task 7 ✅
7. Name 텍스트 탐색(TF-IDF, 누수단어) → Task 8 ✅
8. 누수 후보 점검 → Task 9 ✅
9. 요약 & split 제안 → Task 10 ✅
재사용 유틸 분리(금액 파싱/타깃 이진화) → Task 1 ✅

**Placeholder scan:** 모든 코드 셀에 실제 코드 포함. 마크다운 요약(Task 10 Step 2)은 실행 수치 채움 안내. TBD 없음.

**Type consistency:** `parse_money`, `binarize_target` 시그니처 Task 1 정의와 Task 2 사용 일치. `df["target"]`, `df["is_new"]`, `naics2`, `APPROVAL_FEATURES` 명명 전 태스크 일관.
