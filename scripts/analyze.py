"""Ablation table, error analysis and figures from cached OOF predictions.

Usage:
    python scripts/analyze.py            # analyses FINAL_MODEL (seed-0 OOFs)

Run scripts/validate.py first. Writes results/ablation.csv,
results/error_analysis.json, results/errors_*.csv and results/figures/*.png.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src import cutoff  # noqa: E402
from src.data import load_train  # noqa: E402
from src.pipeline import FINAL_MODEL, load_context, make_clusters  # noqa: E402
from src.validation import TEST_VENUE_MIX, draw_pseudo_test, simulate  # noqa: E402
from src.metric import relevance_coverage  # noqa: E402
from scripts.validate import get_oofs  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"

ABLATION = [
    ("venue_prior", "Metadata only (venue, year, DOI type, #authors)"),
    ("tfidf_lr", "+ title word/char TF-IDF"),
    ("char_lr", "+ title char TF-IDF only"),
    ("emb_nometa_lr", "4 frozen embeddings, no metadata"),
    ("emb_lr", "Metadata + 4 frozen embeddings"),
    ("emb_cluster_lr", "+ KMeans topic clusters"),
    ("emb_author_lr", "+ author target encoding"),
    ("full_lr", "Metadata + TF-IDF + embeddings + clusters"),
    ("emb_ridge", "Binary LR + ridge regression (emb + clusters)"),
    ("gbm", "Gradient boosting on clusters/authors/kNN"),
    ("ens_main", "Ensemble of 6 linear models"),
    ("ens_ft4", "4 fine-tuned encoders"),
    ("ens_final", "Fine-tuned encoders + linear ensemble (final)"),
]


def ablation_table() -> pd.DataFrame:
    rows = []
    for name, desc in ABLATION:
        p = RES / f"cv_{name}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        s = r["strategies"]
        rows.append(
            {
                "model": name,
                "description": desc,
                "auc_worth": r["auc_worth"],
                "spearman_gain": r["spearman_gain"],
                "score_all": s["all"]["score_mean"],
                "score_expected": s["expected"]["score_mean"],
                "Q_expected": s["expected"]["Q_mean"],
                "R_expected": s["expected"]["R_mean"],
                "score_frac_0.6": s["frac_0.6"]["score_mean"],
                "std_expected": s["expected"]["score_std"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RES / "ablation.csv", index=False, float_format="%.4f")
    return df


def tradeoff_curve(train, oof, n_rep=100, seed=7):
    """Mean Q, R, score on pseudo test splits as a function of the kept fraction."""
    fracs = np.round(np.arange(0.3, 1.001, 0.05), 2)
    rng = np.random.default_rng(seed)
    acc = np.zeros((len(fracs), 3))
    for _ in range(n_rep):
        idx = draw_pseudo_test(train, rng)
        meta, pred = train.iloc[idx], oof.iloc[idx]
        perm = rng.permutation(len(idx))
        for j, f in enumerate(fracs):
            prio = cutoff.strategy_fraction(f)(meta, pred)
            for part in (perm[:298], perm[298:]):
                acc[j] += relevance_coverage(meta["id"].values[part], prio[part], meta["Label"].values[part])
    acc /= 2 * n_rep
    return pd.DataFrame({"frac": fracs, "Q": acc[:, 0], "R": acc[:, 1], "score": acc[:, 2]})


def per_group(df, oof, key):
    rows = []
    for g, sub in df.groupby(key):
        o = oof.loc[sub.index]
        y = sub["Label"].values
        rows.append(
            {
                key: g,
                "n": len(sub),
                "worth_rate": float((y > 1).mean()),
                "mean_p_worth": float(o["p_worth"].mean()),
                "auc": float(roc_auc_score(y > 1, o["p_worth"])) if 0 < (y > 1).sum() < len(y) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    abl = ablation_table()
    print(abl[["model", "auc_worth", "score_all", "score_expected", "score_frac_0.6"]].to_string(index=False))

    train, _, store = load_context()
    clusters = make_clusters(store)
    oof = get_oofs(FINAL_MODEL, train, store, clusters, seeds=[0])[0]
    test_like = train["venue"].isin(list(TEST_VENUE_MIX)).values
    tl = train[test_like].copy()
    tl_oof = oof[test_like]

    # Q / R / score versus kept fraction, plus where the adaptive cutoff lands.
    curve = tradeoff_curve(train, oof)
    curve.to_csv(RES / "tradeoff_curve.csv", index=False, float_format="%.4f")
    sim = simulate(train, oof, cutoff.strategy_expected(), n_rep=200)
    kept = sim["n_rec"].mean() / 298
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in ("Q", "R", "score"):
        ax.plot(curve["frac"], curve[c], marker="o", ms=3, label={"Q": "Relevance Q", "R": "Coverage R"}.get(c, "Score"))
    ax.axvline(kept, ls="--", c="gray", label=f"adaptive cutoff (≈{kept:.2f})")
    ax.set_xlabel("fraction of papers recommended")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.set_title(f"{FINAL_MODEL}: relevance / coverage trade-off (CV)")
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff.png", dpi=150)
    plt.close(fig)

    # Ablation bar chart.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(abl["model"], abl["score_expected"], xerr=abl["std_expected"] / np.sqrt(400), color="tab:blue")
    ax.axvline(abl["score_all"].min(), ls=":", c="gray")
    ax.set_xlim(0.5, abl["score_expected"].max() + 0.01)
    ax.set_xlabel("CV score (adaptive cutoff), mean over 400 pseudo-test splits")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png", dpi=150)
    plt.close(fig)

    # Per-venue and per-cluster behaviour.
    venue = per_group(tl, tl_oof, "venue")
    tl["cluster"] = clusters.assign(store.get(clusters.emb_name, tl["id"]))
    clus = per_group(tl, tl_oof, "cluster").sort_values("auc")
    venue.to_csv(RES / "errors_by_venue.csv", index=False, float_format="%.4f")
    clus.to_csv(RES / "errors_by_cluster.csv", index=False, float_format="%.4f")

    # Calibration of P(worthwhile).
    bins = pd.cut(tl_oof["p_worth"], np.linspace(0, 1, 11))
    calib = pd.DataFrame({"p": tl_oof["p_worth"], "y": tl["Label"] > 1}).groupby(bins).agg(
        mean_p=("p", "mean"), rate=("y", "mean"), n=("y", "size")
    )
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], ls=":", c="gray")
    ax.plot(calib["mean_p"], calib["rate"], marker="o")
    ax.set_xlabel("predicted P(rating > 1)")
    ax.set_ylabel("observed share rated > 1")
    ax.set_title("Calibration (OOF, test venues)")
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=150)
    plt.close(fig)

    # Worst individual errors.
    cols = ["id", "venue", "year", "Label", "p_worth", "e_gain", "title"]
    tl["p_worth"], tl["e_gain"] = tl_oof["p_worth"].values, tl_oof["e_gain"].values
    fp = tl[tl["Label"] == 1].nlargest(15, "e_gain")[cols]
    fn = tl[tl["Label"] >= 4].nsmallest(15, "e_gain")[cols]
    fp.to_csv(RES / "errors_top_rated1.csv", index=False, float_format="%.3f")
    fn.to_csv(RES / "errors_excluded_good.csv", index=False, float_format="%.3f")

    by_label = tl.groupby("Label")[["p_worth", "e_gain"]].mean()
    summary = {
        "model": FINAL_MODEL,
        "adaptive_cutoff_kept_fraction": float(kept),
        "best_fixed_fraction": float(curve.loc[curve["score"].idxmax(), "frac"]),
        "best_fixed_fraction_score": float(curve["score"].max()),
        "by_venue": venue.to_dict(orient="records"),
        "mean_pred_by_label": by_label.reset_index().to_dict(orient="records"),
        "worst_clusters": clus.head(5).to_dict(orient="records"),
    }
    (RES / "error_analysis.json").write_text(json.dumps(summary, indent=2, default=float))
    print(venue.to_string(index=False))
    print(by_label.to_string())
    print(f"adaptive cutoff keeps {kept:.3f}; best fixed fraction {summary['best_fixed_fraction']}")


if __name__ == "__main__":
    main()
