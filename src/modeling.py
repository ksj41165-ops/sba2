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
