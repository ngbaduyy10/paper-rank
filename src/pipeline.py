"""Model configurations shared by validation and submission scripts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_test, load_train
from .embeddings import MODELS, embed_titles
from .features import EmbeddingStore, TopicClusters
from .models import GBM, BinaryRidge, Ensemble, FeatureBuilder, OrdinalLR

SEED = 42


def load_context(emb_names=tuple(MODELS)):
    train, test = load_train(), load_test()
    titles = pd.concat([train["title"], test["title"]], ignore_index=True)
    ids = pd.concat([train["id"], test["id"]], ignore_index=True).values
    store = EmbeddingStore(ids, {n: embed_titles(titles, n) for n in emb_names})
    return train, test, store


def make_clusters(store: EmbeddingStore, emb_name: str = "bge", k: int = 30, seed: int = SEED) -> TopicClusters:
    """KMeans fit on all train + test titles (unsupervised, no labels)."""
    tc = TopicClusters(k=k, seed=seed)
    tc.emb_name = emb_name
    tc.fit_kmeans(store.matrices[emb_name])
    return tc


def build(name: str, store: EmbeddingStore, clusters: TopicClusters | None = None):
    """Factory for every named configuration used in the experiments."""
    fb = lambda blocks: FeatureBuilder(blocks, store, clusters)  # noqa: E731
    emb4 = {"emb:bge": 1.0, "emb:specter2": 1.0, "emb:mpnet": 1.0, "emb:minilm": 1.0}

    if name == "venue_prior":
        return OrdinalLR(fb({"meta": 1.0}), C=1.0)
    if name == "tfidf_lr":
        return OrdinalLR(fb({"meta": 1.0, "tfidf": 1.0}), C=1.0)
    if name == "char_lr":
        return OrdinalLR(fb({"meta": 1.0, "char": 1.0}), C=1.0)
    if name == "emb_lr":
        return OrdinalLR(fb({"meta": 1.0, **emb4}), C=0.5)
    if name == "emb_bge_lr":
        return OrdinalLR(fb({"meta": 1.0, "emb:bge": 1.0}), C=2.0)
    if name == "emb_cluster_lr":
        return OrdinalLR(fb({"meta": 1.0, **emb4, "clusters": 1.0}), C=0.5)
    if name == "emb_author_lr":
        return OrdinalLR(fb({"meta": 1.0, **emb4, "clusters": 1.0, "authors": 1.0}), C=0.5)
    if name == "emb_nometa_lr":
        return OrdinalLR(fb(emb4), C=0.5)
    if name == "full_lr":
        return OrdinalLR(fb({"meta": 1.0, "tfidf": 1.0, **emb4, "clusters": 1.0}), C=0.5)
    if name == "emb_ridge":
        return BinaryRidge(fb({"meta": 1.0, **emb4, "clusters": 1.0}), C=1.0, alpha=3.0)
    if name == "tfidf_ridge":
        return BinaryRidge(fb({"meta": 1.0, "tfidf": 1.0}), C=2.0, alpha=1.0)
    if name == "gbm":
        return GBM(
            fb({"meta": 1.0, "clusters": 1.0, "authors": 1.0, "knn:bge": 1.0, "knn:specter2": 1.0, "knn:mpnet": 1.0}),
            seed=SEED,
        )
    if name.startswith("emblr_"):
        return OrdinalLR(fb({"meta": 1.0, "emb:" + name[6:]: 1.0}), C=3.0)
    if name.startswith("embclr_"):
        return OrdinalLR(fb({"meta": 1.0, "emb:" + name[7:]: 1.0, "clusters": 1.0}), C=3.0)
    if name.startswith("embridge_"):
        return BinaryRidge(fb({"meta": 1.0, "emb:" + name[9:]: 1.0, "clusters": 1.0}), C=3.0, alpha=1.0)
    if name in FINETUNE:
        from .finetune import FineTuned

        return FineTuned(**FINETUNE[name])
    if name.startswith("ens"):
        return Ensemble([build(m, store, clusters) for m in ENSEMBLES[name]], ENSEMBLE_WEIGHTS.get(name))
    raise KeyError(name)


FINETUNE = {
    "ft_bge": dict(repo="BAAI/bge-base-en-v1.5", epochs=3, lr=3e-5, seed=0),
    "ft_bge_s1": dict(repo="BAAI/bge-base-en-v1.5", epochs=3, lr=3e-5, seed=1),
    "ft_bge_ep5": dict(repo="BAAI/bge-base-en-v1.5", epochs=5, lr=2e-5, seed=0),
    "ft_e5": dict(repo="intfloat/e5-base-v2", epochs=3, lr=3e-5, seed=0),
    "ft_scibert": dict(repo="allenai/scibert_scivocab_uncased", epochs=3, lr=3e-5, seed=0),
    "ft_mpnet": dict(repo="sentence-transformers/all-mpnet-base-v2", epochs=3, lr=3e-5, seed=0),
    "ft_bge_large": dict(repo="BAAI/bge-large-en-v1.5", epochs=3, lr=1.5e-5, batch_size=16, seed=0),
}

ENSEMBLES = {
    "ens_main": ["char_lr", "tfidf_lr", "emb_lr", "emb_cluster_lr", "emb_ridge", "tfidf_ridge"],
    "ens_final": ["ft_bge", "ft_e5", "ft_mpnet", "ft_scibert", "ens_main"],
    "ens_ft4": ["ft_bge", "ft_e5", "ft_mpnet", "ft_scibert"],
}
FINAL_MODEL = "ens_final"
FINAL_STRATEGY = "expected"
FINAL_EMBEDDINGS = ("bge", "specter2", "mpnet", "minilm")
ENSEMBLE_WEIGHTS: dict[str, list[float]] = {}


def fit_predict_fn(name: str, store: EmbeddingStore, clusters: TopicClusters | None):
    def f(tr_df, va_df):
        model = build(name, store, clusters)
        return model.fit(tr_df).predict(va_df)
    return f


def rank01(x) -> np.ndarray:
    return pd.Series(np.asarray(x)).rank(pct=True).values
