"""Exploratory data analysis: summary statistics and report figures.

Usage:
    python scripts/eda.py

Writes results/eda.json and results/figures/eda_*.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import load_test, load_train, split_authors  # noqa: E402
from src.metric import relevance_coverage  # noqa: E402
from src.validation import TEST_VENUE_MIX, draw_pseudo_test, title_group  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"


def metric_simulations(train, n_rep=300, seed=0):
    """Score of simple oracle / naive strategies on test-mix pseudo splits."""
    rng = np.random.default_rng(seed)
    out = {"random_all": [], "perfect_order_all": [], "oracle_exclude_rating1": [], "oracle_exclude_and_order": []}
    for _ in range(n_rep):
        part = rng.permutation(draw_pseudo_test(train, rng))[:298]
        d = train.iloc[part]
        ids, y = d["id"].values, d["Label"].values
        noise = rng.random(len(y))
        out["random_all"].append(relevance_coverage(ids, 1 + noise, y)[2])
        out["perfect_order_all"].append(relevance_coverage(ids, y + noise * 0.1, y)[2])
        out["oracle_exclude_rating1"].append(relevance_coverage(ids, np.where(y > 1, 1 + noise, 0), y)[2])
        out["oracle_exclude_and_order"].append(relevance_coverage(ids, np.where(y > 1, y, 0), y)[2])
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    train, test = load_train(), load_test()
    train["n_auth"] = train["authors"].map(lambda s: len(split_authors(s)))
    train["worth"] = train["Label"] > 1

    by_venue = train.groupby("venue").agg(
        n=("Label", "size"), mean_label=("Label", "mean"), worth_rate=("worth", "mean")
    )
    by_venue["n_test"] = test["venue"].value_counts().reindex(by_venue.index).fillna(0).astype(int)
    by_year = train.groupby("year").agg(n=("Label", "size"), worth_rate=("worth", "mean"), mean_label=("Label", "mean"))
    test_authors = {a for s in test["authors"] for a in split_authors(s)}
    train_authors = {a for s in train["authors"] for a in split_authors(s)}
    test_mentions = [a for s in test["authors"] for a in split_authors(s)]

    w = pd.Series(TEST_VENUE_MIX) / sum(TEST_VENUE_MIX.values())
    test_like_worth = float((by_venue.loc[w.index, "worth_rate"] * w).sum())

    stats = {
        "n_train": len(train),
        "n_test": len(test),
        "label_counts": train["Label"].value_counts().sort_index().to_dict(),
        "by_venue": by_venue.round(4).reset_index().to_dict(orient="records"),
        "by_year": by_year.round(4).reset_index().to_dict(orient="records"),
        "no_author_worth_rate": float(train.loc[train["n_auth"] == 0, "worth"].mean()),
        "with_author_worth_rate": float(train.loc[train["n_auth"] > 0, "worth"].mean()),
        "n_no_author": int((train["n_auth"] == 0).sum()),
        "duplicate_title_rows": int(pd.Series(title_group(train["title"])).duplicated(keep=False).sum()),
        "test_author_mentions": len(test_mentions),
        "test_author_mentions_seen_in_train": int(sum(a in train_authors for a in test_mentions)),
        "distinct_test_authors_seen_in_train": f"{len(test_authors & train_authors)}/{len(test_authors)}",
        "title_len_words_mean": float(train["title"].str.split().str.len().mean()),
        "corr_id_year_within_venue": train.groupby("venue").apply(lambda d: d["id"].corr(d["year"])).round(3).to_dict(),
        "test_mix_expected_worth_rate": test_like_worth,
        "metric_simulations": metric_simulations(train),
    }
    (RES / "eda.json").write_text(json.dumps(stats, indent=2, default=float))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ct = pd.crosstab(train["venue"], train["Label"], normalize="index")
    ct.plot.barh(stacked=True, ax=axes[0], colormap="RdYlGn", width=0.8)
    axes[0].set_title("Rating distribution by venue (train)")
    axes[0].set_xlabel("share")
    axes[0].legend(title="rating", ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.35), fontsize=8)
    mix = pd.DataFrame(
        {"train": train["venue"].value_counts(normalize=True), "test": test["venue"].value_counts(normalize=True)}
    ).fillna(0)
    mix.plot.bar(ax=axes[1])
    axes[1].set_title("Venue mix: train vs test")
    axes[1].set_ylabel("share of papers")
    for v, d in train.groupby("venue"):
        g = d.groupby("year")["worth"].mean()
        axes[2].plot(g.index, g.values, marker="o", ms=3, label=v)
    axes[2].set_title("Share rated > 1 by year")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "eda_overview.png", dpi=150)
    plt.close(fig)

    print(json.dumps({k: stats[k] for k in ("label_counts", "test_mix_expected_worth_rate", "metric_simulations")}, indent=1))
    print(by_venue.round(3).to_string())


if __name__ == "__main__":
    main()
