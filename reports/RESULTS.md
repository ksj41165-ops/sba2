# SBA 부실예측 — 결과 보고서

> 정형 + 비정형(사업체명) 데이터를 활용한 씬파일러 부실여부 예측
> 데이터: `SBAnational.csv` (899,164건) · 작성일 2026-05-30

---

## 1. 문제 정의

- **목표:** 대출 **승인 시점**에 부실(CHGOFF) 여부를 예측한다.
- **타깃:** `MIS_Status` → `CHGOFF`=1(부실) / `P I F`=0(정상)
- **씬파일러:** 신용이력 컬럼이 없어 직접 식별 불가 → **신규사업자(`NewExist=2`)를 thin-file proxy**로 보고 별도 리포팅. (신용이력 부족 ≈ 사업운영이력 부족의 근사임을 명시)

## 2. 데이터 처리 핵심 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| 예측 시점 | **승인 시점** | 심사모델 관점. 승인 이후 관측값은 사용 불가 |
| 누수 제외 | `DisbursementGross/Date`, `BalanceGross`, `ChgOffDate`, `ChgOffPrinGr` | 부실 결과 이후 확정되는 값 |
| 시간 분할 | train ≤2006 / valid 2007 / test 2008–2010 | 시계열 일반화 평가 |
| 절단 보정 | **2011년 이후 제외** | 최근 대출은 만기 미도래로 부실 과소집계(2013년 2.9%·2,455건) |
| 불균형 | 부실률 17.6% → `class_weight`/`scale_pos_weight` | PR-AUC 핵심 지표 사용 |

**분할 결과:** train 727,785건(부실 14.1%) · valid 71,639건(42.8%) · test 75,296건(29.9%)
> test 코호트(2008–2010)는 금융위기 영향으로 부실률이 높아 **보수적 평가**가 됨.

## 3. 피처

- **정형(수치):** GrAppv, SBA_Appv, **sba_ratio(=SBA_Appv/GrAppv)**, Term, NoEmp, CreateJob, RetainedJob
- **정형(범주):** NewExist, is_franchise, UrbanRural, RevLineCr, LowDoc, State, BankState, **naics2(업종 2자리)**
- **비정형:** 사업체명 `Name` → **TF-IDF (char n-gram 3–5, min_df=20, max_features=5000)**
- 전처리(스케일/원핫/TF-IDF)는 **train에만 fit** 후 test에 transform (누수 차단)

## 4. 모델 성능 (test = 2008–2010)

| 모델 | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| logreg \| 정형 | 0.5032 | 0.7502 | 0.4269 | 0.8569 | 0.5699 |
| logreg \| 정형+Name | 0.5150 | 0.7550 | 0.4347 | 0.8374 | 0.5723 |
| lgbm \| 정형 | 0.8882 | 0.9467 | 0.7473 | 0.8992 | 0.8162 |
| **lgbm \| 정형+Name** | **0.8896** | **0.9496** | 0.7432 | 0.9006 | 0.8144 |

원본: [`modeling_results.csv`](modeling_results.csv) · 곡선: [`figures/pr_curves.png`](figures/pr_curves.png)

### 해석
- **LightGBM이 로지스틱을 압도** (PR-AUC 0.89 vs 0.50). 표 형식 혼합 데이터에 트리 부스팅이 적합.
- **비정형(Name TF-IDF) 효과는 소폭 개선** (정형 0.8882 → 정형+Name 0.8896). 약한 신호이며, 큰 폭 상승이 아니라는 점이 오히려 **누수가 없음**을 시사.
- 높은 성능은 승인시점 신호(Term, 보증비율 sba_ratio, 대출금액, 업종 등)로 설명되며, 이는 공개 SBA 연구 결과와 일치.

## 5. 씬파일러(thin-file proxy) 세그먼트 — 최고 모델 기준

| 세그먼트 | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 전체 | 0.8896 | 0.9496 | 0.7432 | 0.9006 | 0.8144 |
| **신규사업자(thin-file)** | 0.8788 | 0.9523 | 0.7337 | 0.9238 | 0.8179 |
| 기존사업자 | 0.8933 | 0.9485 | 0.7480 | 0.8896 | 0.8127 |

원본: [`segment_results.csv`](segment_results.csv)

- **씬파일러 세그먼트에서도 모델이 잘 작동** (PR-AUC 0.879, recall 0.924). 기존사업자 대비 PR-AUC는 약간 낮지만 recall은 더 높음.

## 6. 한계 및 향후 과제

- **valid셋 미사용:** 고정 하이퍼파라미터를 써서 valid(2007)는 분할만 함 → 향후 튜닝에 활용.
- **씬파일러는 proxy:** 실제 신용이력 데이터가 아닌 신규사업자 근사. 본 데이터의 구조적 한계.
- **시간 변동성:** 2006~2008 위기 코호트의 극단적 부실률 → 거시환경 피처 보강 여지.
- **향후:** 임계값 최적화, 피처 중요도/SHAP 분석, 외부 비정형 데이터 결합, 딥러닝 텍스트 인코더 실험.

## 7. 재현 방법

```bash
python -m pytest -q                       # 모듈 단위테스트 8건
PYTHONUTF8=1 jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3000 notebooks/01_sba_eda.ipynb
PYTHONUTF8=1 jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3000 notebooks/02_sba_modeling.ipynb
```

관련 문서: 설계 `docs/superpowers/specs/2026-05-30-sba-default-prediction-guide-design.md` · 계획 `docs/superpowers/plans/`
