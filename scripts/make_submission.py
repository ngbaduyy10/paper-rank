"""Fit the selected model on all of train and write the Kaggle submission.

Usage:
    python scripts/make_submission.py                       # FINAL_MODEL / FINAL_STRATEGY
    python scripts/make_submission.py --model ens_main --out submissions/ens_main.csv

Writes submissions/<name>.csv (id,priority) and submissions/<name>_pred.csv
(raw p_worth / e_gain per test paper, for the report).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import load_submission_ids  # noqa: E402
from src.pipeline import (  # noqa: E402
    FINAL_EMBEDDINGS,
    FINAL_MODEL,
    FINAL_STRATEGY,
    SEED,
    build,
    load_context,
    make_clusters,
)
from scripts.validate import STRATEGIES  # noqa: E402


def check_submission(sub: pd.DataFrame, expected_ids: list[int]) -> None:
    assert list(sub.columns) == ["id", "priority"]
    assert len(sub) == len(expected_ids) == sub["id"].nunique()
    assert set(sub["id"]) == set(expected_ids)
    assert np.isfinite(sub["priority"]).all() and (sub["priority"] >= 0).all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=FINAL_MODEL)
    ap.add_argument("--strategy", default=FINAL_STRATEGY, choices=list(STRATEGIES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    np.random.seed(SEED)
    t0 = time.time()
    train, test, store = load_context(FINAL_EMBEDDINGS) if args.model == FINAL_MODEL else load_context()
    clusters = make_clusters(store)
    model = build(args.model, store, clusters).fit(train)
    pred = model.predict(test)[["p_worth", "e_gain"]].reset_index(drop=True)
    prio = STRATEGIES[args.strategy](test.reset_index(drop=True), pred)

    sub = pd.DataFrame({"id": test["id"].values, "priority": prio})
    ids = load_submission_ids()
    sub = sub.set_index("id").loc[ids].reset_index()
    check_submission(sub, ids)

    out = Path(args.out) if args.out else ROOT / "submissions" / f"{args.model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    pred.insert(0, "id", test["id"].values)
    pred.to_csv(out.with_name(out.stem + "_pred.csv"), index=False)

    n_rec = int((sub["priority"] > 0).sum())
    print(
        f"{args.model}/{args.strategy}: wrote {out.relative_to(ROOT)} "
        f"({n_rec}/{len(sub)} recommended, mean P(worth)={pred['p_worth'].mean():.3f}) [{time.time() - t0:.0f}s]"
    )


if __name__ == "__main__":
    main()
