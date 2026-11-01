"""Cross-validation and test-like simulation of the hidden Public/Private splits.

Protocol
--------
1. Out-of-fold (OOF) predictions are produced with repeated StratifiedGroupKFold
   on train (strata = venue x label, groups = normalised title so duplicates
   never straddle folds).
2. A pseudo test set of 596 papers is drawn (without replacement) from the OOF
   rows so that its venue mix equals the real test venue mix.
3. The submission strategy (scores -> priorities, including the priority-0
   cutoff) is applied to the whole pseudo test set, exactly as it would be
   applied to the real test file.
4. The pseudo test set is randomly halved into two 298-paper splits that are
   scored independently with the competition metric.
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .metric import relevance_coverage

TEST_VENUE_MIX = {"cav": 221, "lics": 187, "kr": 151, "lpnmr": 37}


def title_group(titles: pd.Series) -> np.ndarray:
    norm = titles.str.lower().map(lambda s: re.sub(r"[^a-z0-9]+", " ", s).strip())
    return pd.factorize(norm)[0]


def cv_splits(df: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    strata = df["venue"] + "_" + df["Label"].astype(str)
    groups = title_group(df["title"])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(df, strata, groups))


def oof_predict(df: pd.DataFrame, fit_predict: Callable, n_splits: int = 5, seeds=(0, 1, 2)) -> list[pd.DataFrame]:
    """Run fit_predict(train_df, valid_df) -> DataFrame of score columns, per repeat.

    Returns one OOF frame per seed, aligned with df.index.
    """
    outs = []
    for seed in seeds:
        parts = []
        for tr_idx, va_idx in cv_splits(df, n_splits, seed):
            pred = fit_predict(df.iloc[tr_idx], df.iloc[va_idx])
            pred.index = df.index[va_idx]
            parts.append(pred)
        outs.append(pd.concat(parts).loc[df.index])
    return outs


def draw_pseudo_test(df: pd.DataFrame, rng: np.random.Generator, mix=TEST_VENUE_MIX) -> np.ndarray:
    idx = []
    for venue, n in mix.items():
        pool = np.flatnonzero(df["venue"].values == venue)
        idx.append(rng.choice(pool, size=min(n, len(pool)), replace=False))
    return np.concatenate(idx)


def simulate(
    df: pd.DataFrame,
    oof: pd.DataFrame,
    strategy: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    n_rep: int = 200,
    seed: int = 123,
) -> pd.DataFrame:
    """Score a strategy on repeated pseudo test sets; one row per 298-paper split.

    strategy(meta_df, pred_df) -> priority array for the rows of meta_df.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(n_rep):
        idx = draw_pseudo_test(df, rng)
        meta = df.iloc[idx]
        pred = oof.iloc[idx]
        prio = np.asarray(strategy(meta, pred), dtype=float)
        perm = rng.permutation(len(idx))
        half = len(idx) // 2
        for split, part in (("A", perm[:half]), ("B", perm[half:])):
            q, r, s = relevance_coverage(meta["id"].values[part], prio[part], meta["Label"].values[part])
            rows.append({"rep": rep, "split": split, "Q": q, "R": r, "score": s, "n_rec": int((prio[part] > 0).sum())})
    return pd.DataFrame(rows)


def summarize(sim: pd.DataFrame) -> dict:
    return {
        "score_mean": float(sim["score"].mean()),
        "score_std": float(sim["score"].std()),
        "score_p05": float(sim["score"].quantile(0.05)),
        "score_p95": float(sim["score"].quantile(0.95)),
        "Q_mean": float(sim["Q"].mean()),
        "R_mean": float(sim["R"].mean()),
        "n_rec_mean": float(sim["n_rec"].mean()),
    }


def evaluate(df, oofs, strategy, n_rep=200) -> dict:
    """Average the simulation over every CV repeat."""
    sims = [simulate(df, oof, strategy, n_rep=n_rep, seed=100 + i) for i, oof in enumerate(oofs)]
    sim = pd.concat(sims, ignore_index=True)
    out = summarize(sim)
    out["repeat_means"] = [float(s["score"].mean()) for s in sims]
    return out
