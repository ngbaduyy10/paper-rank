"""Out-of-fold validation of model configurations with the competition metric.

Usage:
    python scripts/validate.py --models venue_prior tfidf_lr ens_main
    python scripts/validate.py --models ens_main --n-rep 300 --seeds 0 1 2

Writes results/oof/<model>_s<seed>.csv and results/cv_<model>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src import cutoff  # noqa: E402
from src.pipeline import ENSEMBLE_WEIGHTS, ENSEMBLES, fit_predict_fn, load_context, make_clusters  # noqa: E402
from src.validation import cv_splits, evaluate  # noqa: E402

OOF_DIR = ROOT / "results" / "oof"


def get_oofs(name, train, store, clusters, seeds, n_splits=5, refresh=False):
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    if name in ENSEMBLES and not refresh:
        # Ensemble OOF = weighted mean of member OOFs on the same folds.
        members = ENSEMBLES[name]
        w = np.asarray(ENSEMBLE_WEIGHTS.get(name, np.ones(len(members))), float)
        w = w / w.sum()
        per_member = [get_oofs(m, train, store, clusters, seeds, n_splits) for m in members]
        return [
            sum(wi * oofs[i][["p_worth", "e_gain"]] for wi, oofs in zip(w, per_member)) for i in range(len(seeds))
        ]
    f = fit_predict_fn(name, store, clusters)
    oofs = []
    for seed in seeds:
        path = OOF_DIR / f"{name}_s{seed}.csv"
        if path.exists() and not refresh:
            oofs.append(pd.read_csv(path, index_col=0))
            continue
        parts = []
        for tr_idx, va_idx in cv_splits(train, n_splits, seed):
            pred = f(train.iloc[tr_idx], train.iloc[va_idx])
            pred.index = train.index[va_idx]
            parts.append(pred)
        oof = pd.concat(parts).loc[train.index]
        oof.to_csv(path)
        oofs.append(oof)
    return oofs


def ranking_stats(train, oofs):
    y = train["Label"].values
    test_like = (train["venue"] != "iclp").values
    auc = [roc_auc_score(y[test_like] > 1, o["p_worth"].values[test_like]) for o in oofs]
    rho = [spearmanr(o["e_gain"].values[test_like], y[test_like]).correlation for o in oofs]
    return {"auc_worth": float(np.mean(auc)), "spearman_gain": float(np.mean(rho))}


STRATEGIES = {
    "all": cutoff.strategy_all(),
    "expected": cutoff.strategy_expected(),
    "expected_x1.1": cutoff.strategy_expected(1.1),
    "frac_0.6": cutoff.strategy_fraction(0.6),
    "frac_0.7": cutoff.strategy_fraction(0.7),
    "frac_0.8": cutoff.strategy_fraction(0.8),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-rep", type=int, default=200)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    args = ap.parse_args()

    train, test, store = load_context()
    clusters = make_clusters(store)
    for name in args.models:
        t0 = time.time()
        oofs = get_oofs(name, train, store, clusters, args.seeds, refresh=args.refresh)
        res = {"model": name, **ranking_stats(train, oofs), "strategies": {}}
        for s in args.strategies:
            res["strategies"][s] = evaluate(train, oofs, STRATEGIES[s], n_rep=args.n_rep)
        res["seconds"] = round(time.time() - t0, 1)
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / f"cv_{name}.json").write_text(json.dumps(res, indent=2))
        line = " ".join(
            f"{s}={v['score_mean']:.4f}(Q{v['Q_mean']:.3f}/R{v['R_mean']:.3f})" for s, v in res["strategies"].items()
        )
        print(f"{name:16} auc={res['auc_worth']:.4f} rho={res['spearman_gain']:.4f} {line} [{res['seconds']}s]", flush=True)


if __name__ == "__main__":
    main()
