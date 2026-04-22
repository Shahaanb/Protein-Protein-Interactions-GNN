from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true.astype(int) == y_pred.astype(int))) if y_true.size else 0.0


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    c = confusion_counts(y_true, y_pred)
    p = _safe_div(c["tp"], (c["tp"] + c["fp"]))
    r = _safe_div(c["tp"], (c["tp"] + c["fn"]))
    f1 = _safe_div(2.0 * p * r, (p + r))
    return {"precision": p, "recall": r, "f1": f1}


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # AUC via rank statistic; handles ties via average rank.
    y_true = y_true.astype(int)
    y_score = y_score.astype(float)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    # average ranks for ties
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg_rank = float(np.mean(ranks[order[i:j]]))
            ranks[order[i:j]] = avg_rank
        i = j

    sum_ranks_pos = float(np.sum(ranks[pos]))
    u = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(u / (n_pos * n_neg))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Area under the precision-recall curve.
    y_true = y_true.astype(int)
    y_score = y_score.astype(float)

    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = 0
    fp = 0
    precisions: List[float] = []
    recalls: List[float] = []

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(_safe_div(tp, tp + fp))
        recalls.append(_safe_div(tp, n_pos))

    # Add (0,1) start point
    recalls = [0.0] + recalls
    precisions = [1.0] + precisions

    # Integrate precision over recall
    area = 0.0
    for i in range(1, len(recalls)):
        dr = recalls[i] - recalls[i - 1]
        area += precisions[i] * dr
    return float(area)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(float)
    y_prob = y_prob.astype(float)
    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = y_true.astype(float)
    y_prob = y_prob.astype(float)
    if y_true.size == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        if b1 == 1.0:
            mask = (y_prob >= b0) & (y_prob <= b1)
        else:
            mask = (y_prob >= b0) & (y_prob < b1)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        w = float(np.mean(mask))
        ece += w * abs(acc - conf)
    return float(ece)


def find_best_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    if y_true.size == 0:
        return {"threshold": 0.5, "f1": 0.0}

    # Candidate thresholds: unique probs + endpoints.
    thresholds = np.unique(y_prob)
    if thresholds.size > 512:
        # downsample to keep it fast
        thresholds = np.quantile(y_prob, np.linspace(0.0, 1.0, 512))

    best = {"threshold": 0.5, "f1": -1.0}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = precision_recall_f1(y_true, y_pred)["f1"]
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1)}

    return best


def compute_all_metrics(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (y_prob >= float(threshold)).astype(int)

    c = confusion_counts(y_true, y_pred)
    prf = precision_recall_f1(y_true, y_pred)

    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy(y_true, y_pred)),
        "precision": float(prf["precision"]),
        "recall": float(prf["recall"]),
        "f1": float(prf["f1"]),
        "tp": float(c["tp"]),
        "tn": float(c["tn"]),
        "fp": float(c["fp"]),
        "fn": float(c["fn"]),
        "roc_auc": float(roc_auc(y_true, y_prob)),
        "pr_auc": float(pr_auc(y_true, y_prob)),
        "brier": float(brier_score(y_true, y_prob)),
        "ece_10": float(expected_calibration_error(y_true, y_prob, n_bins=10)),
    }
    return out


def bootstrap_ci(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int = 200,
    seed: int = 1337,
) -> Dict[str, Tuple[float, float]]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    n = int(y_true.size)
    if n == 0:
        return {}

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if pos_idx.size == 0 or neg_idx.size == 0:
        # Can't form meaningful CIs for discrimination/calibration metrics with a single class.
        return {}

    rng = np.random.default_rng(seed)

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": [],
        "brier": [],
        "ece_10": [],
    }

    n_pos = int(pos_idx.size)
    n_neg = int(neg_idx.size)

    # Stratified bootstrap: sample positives and negatives independently.
    for _ in range(int(n_boot)):
        idx_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        idx_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([idx_pos, idx_neg])
        rng.shuffle(idx)
        yt = y_true[idx]
        yp = y_prob[idx]
        m = compute_all_metrics(y_true=yt, y_prob=yp, threshold=threshold)
        for k in metrics.keys():
            metrics[k].append(float(m[k]))

    cis: Dict[str, Tuple[float, float]] = {}
    for k, vals in metrics.items():
        arr = np.asarray(vals, dtype=float)
        lo = float(np.quantile(arr, 0.025))
        hi = float(np.quantile(arr, 0.975))
        cis[k] = (lo, hi)

    return cis
