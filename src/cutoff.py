"""Turn model scores into submission priorities (ordering + priority-0 cutoff)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_priority(order_score: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Priorities in (0, 10] for selected papers (higher = earlier), 0 otherwise."""
    order_score = np.asarray(order_score, dtype=float)
    ranks = pd.Series(order_score).rank(method="first").values
    prio = 10.0 * ranks / len(ranks)
    return np.where(selected, np.round(prio, 6), 0.0)


def expected_score_curve(e_gain: np.ndarray, p_worth: np.ndarray, order_score: np.ndarray | None = None) -> np.ndarray:
    """Plug-in expected score of recommending the top-k papers, for k = 1..n.

    Papers are read in descending order_score (default: e_gain). Q_k uses the
    expected gain of each position, R_k = sum of P(worthwhile) of the selected
    papers / expected number of worthwhile papers.
    """
    order_score = e_gain if order_score is None else order_score
    o = np.argsort(-np.asarray(order_score), kind="stable")
    g = np.asarray(e_gain)[o]
    p = np.asarray(p_worth)[o]
    w = 1.0 / np.log2(np.arange(1, len(o) + 1) + 1)
    q = np.cumsum(w * g) / np.cumsum(w)
    r = np.cumsum(p) / max(p.sum(), 1e-12)
    return np.where(q + r > 0, 2 * q * r / np.maximum(q + r, 1e-12), 0.0)


def select_expected(e_gain, p_worth, order_score=None, shrink: float = 1.0) -> np.ndarray:
    """Keep the top-k that maximises the plug-in expected score.

    shrink > 1 keeps proportionally more papers (hedge against over-cutting).
    """
    order_score = e_gain if order_score is None else order_score
    curve = expected_score_curve(e_gain, p_worth, order_score)
    k = int(np.argmax(curve)) + 1
    k = min(len(curve), int(round(k * shrink)))
    o = np.argsort(-np.asarray(order_score), kind="stable")
    selected = np.zeros(len(curve), dtype=bool)
    selected[o[:k]] = True
    return selected


def select_fraction(order_score, frac: float) -> np.ndarray:
    n = len(order_score)
    k = int(round(frac * n))
    o = np.argsort(-np.asarray(order_score), kind="stable")
    selected = np.zeros(n, dtype=bool)
    selected[o[:k]] = True
    return selected


def select_threshold(p_worth, t: float) -> np.ndarray:
    return np.asarray(p_worth) >= t


# Strategies: (meta_df, pred_df) -> priority array. pred_df has e_gain, p_worth.

def strategy_all(order_col: str = "e_gain"):
    def f(meta, pred):
        return to_priority(pred[order_col].values, np.ones(len(pred), dtype=bool))
    return f


def strategy_expected(shrink: float = 1.0, order_col: str = "e_gain"):
    def f(meta, pred):
        sel = select_expected(pred["e_gain"].values, pred["p_worth"].values, pred[order_col].values, shrink)
        return to_priority(pred[order_col].values, sel)
    return f


def strategy_fraction(frac: float, order_col: str = "e_gain"):
    def f(meta, pred):
        return to_priority(pred[order_col].values, select_fraction(pred[order_col].values, frac))
    return f


def strategy_threshold(t: float, order_col: str = "e_gain"):
    def f(meta, pred):
        return to_priority(pred[order_col].values, select_threshold(pred["p_worth"].values, t))
    return f
