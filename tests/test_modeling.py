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
    assert out["new(thin-file)"]["recall"] == 1.0
