"""Exact re-implementation of the competition metric."""
from __future__ import annotations

import numpy as np


def reading_order(ids, priority) -> np.ndarray:
    """Indices of recommended papers (priority > 0) in evaluator reading order.

    Descending priority; ties broken by ascending id compared as text.
    """
    ids = np.asarray(ids)
    priority = np.asarray(priority, dtype=float)
    keep = np.flatnonzero(priority > 0)
    keys = sorted(keep, key=lambda i: (-priority[i], str(ids[i])))
    return np.asarray(keys, dtype=int)


def relevance_coverage(ids, priority, labels) -> tuple[float, float, float]:
    """Return (Q, R, score) for one evaluation split."""
    labels = np.asarray(labels)
    order = reading_order(ids, priority)
    total_worth = int((labels > 1).sum())
    if len(order) == 0 or total_worth == 0:
        return 0.0, 0.0, 0.0
    y = labels[order]
    w = 1.0 / np.log2(np.arange(1, len(order) + 1) + 1)
    q = float((w * (y - 1) / 4.0).sum() / w.sum())
    r = float((y > 1).sum() / total_worth)
    s = 0.0 if q == 0 or r == 0 else 2 * q * r / (q + r)
    return q, r, s


def score(ids, priority, labels) -> float:
    return relevance_coverage(ids, priority, labels)[2]
