"""Metrics for an ORDINAL four-grade target.

Accuracy alone treats "Mayo 0 called Mayo 3" like "Mayo 0 called Mayo 1". The main metric here is the
quadratic weighted kappa (QWK), which punishes far errors more. The report also gives macro-F1,
per-grade recall, the confusion matrix, the mean absolute grade error, the remission-versus-active
AUC and the calibration error, each with a bootstrap interval where it is cheap.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

K = 4


def qwk(y_true, y_pred, k: int = K) -> float:
    """Quadratic weighted Cohen's kappa."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    obs = np.zeros((k, k))
    np.add.at(obs, (y_true, y_pred), 1)
    w = np.subtract.outer(np.arange(k), np.arange(k)) ** 2 / (k - 1) ** 2
    expected = np.outer(obs.sum(1), obs.sum(0)) / max(obs.sum(), 1)
    denom = (w * expected).sum()
    return float(1 - (w * obs).sum() / denom) if denom > 0 else 0.0


def expected_grade(proba: np.ndarray) -> np.ndarray:
    return np.asarray(proba) @ np.arange(K)


def ece(y_true, proba, bins: int = 10) -> float:
    proba = np.asarray(proba)
    conf, pred = proba.max(1), proba.argmax(1)
    correct = pred == np.asarray(y_true)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def bootstrap(metric, y_true, y_pred, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = [metric(y_true[i], y_pred[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def report(y_true, proba, n_boot: int = 1000, seed: int = 0) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    y_pred = proba.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.full(K, np.nan), where=cm.sum(1) > 0)
    out = {
        "n": int(len(y_true)),
        "support": cm.sum(1).tolist(),
        "qwk": qwk(y_true, y_pred),
        "qwk_ci95": list(bootstrap(qwk, y_true, y_pred, n_boot, seed)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(K)), average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "per_grade_recall": [None if np.isnan(r) else float(r) for r in recall],
        "mae_grades": float(np.abs(y_true - y_pred).mean()),
        "confusion": cm.tolist(),
        "ece": ece(y_true, proba),
    }
    active_true = (y_true >= 2).astype(int)
    if 0 < active_true.sum() < len(active_true):
        p_active = proba[:, 2:].sum(1)
        out["remission_vs_active_auc"] = float(roc_auc_score(active_true, p_active))
    return out
