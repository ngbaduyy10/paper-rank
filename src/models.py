"""Base models and the ensemble that produces P(worthwhile) and expected gain.

Every model exposes fit(train_df) and predict(df) -> DataFrame with columns
  p_worth : P(rating > 1)            -> classification
  e_gain  : E[(rating - 1) / 4]      -> regression / ordinal expectation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression, Ridge

from .features import (
    AuthorTargetEncoder,
    EmbeddingStore,
    MetaOneHot,
    TitleTfidf,
    TopicClusters,
    gain,
    knn_gain,
)

GAINS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


class FeatureBuilder:
    """Sparse design matrix from weighted feature blocks.

    blocks: dict name -> weight. Names: 'meta', 'tfidf', 'word', 'char',
    'emb:<model>', 'clusters', 'authors', 'knn:<model>'.
    """

    def __init__(self, blocks: dict, store: EmbeddingStore, clusters: TopicClusters | None = None, knn_k: int = 25):
        self.blocks, self.store, self.clusters, self.knn_k = blocks, store, clusters, knn_k

    def fit_transform(self, df):
        self.parts = {}
        mats = []
        for name, w in self.blocks.items():
            if name == "meta":
                self.parts[name] = MetaOneHot().fit(df)
                m = self.parts[name].transform(df)
            elif name in ("tfidf", "word", "char"):
                self.parts[name] = TitleTfidf(word=name != "char", char=name != "word").fit(df)
                m = self.parts[name].transform(df)
            elif name.startswith("emb:"):
                m = self.store.get(name[4:], df["id"])
            elif name == "clusters":
                emb = self.store.get(self.clusters.emb_name, df["id"])
                self.clusters.fit(emb, df["Label"].values)
                m = self.clusters.transform(emb)
            elif name == "authors":
                self.parts[name] = AuthorTargetEncoder()
                m = self.parts[name].fit_transform(df)
            elif name.startswith("knn:"):
                emb = self.store.get(name[4:], df["id"])
                self.parts[name] = (emb, df["Label"].values)
                m = knn_gain(emb, df["Label"].values, emb, self.knn_k + 1, exclude_self=True)
            else:
                raise KeyError(name)
            mats.append(_as_sparse(m) * w)
        return sp.hstack(mats).tocsr()

    def transform(self, df):
        mats = []
        for name, w in self.blocks.items():
            if name in ("meta", "tfidf", "word", "char"):
                m = self.parts[name].transform(df)
            elif name.startswith("emb:"):
                m = self.store.get(name[4:], df["id"])
            elif name == "clusters":
                m = self.clusters.transform(self.store.get(self.clusters.emb_name, df["id"]))
            elif name == "authors":
                m = self.parts[name].transform(df)
            elif name.startswith("knn:"):
                emb_tr, lab = self.parts[name]
                m = knn_gain(emb_tr, lab, self.store.get(name[4:], df["id"]), self.knn_k)
            mats.append(_as_sparse(m) * w)
        return sp.hstack(mats).tocsr()


def _as_sparse(m):
    return m if sp.issparse(m) else sp.csr_matrix(np.asarray(m, dtype=float))


class OrdinalLR:
    """Multinomial logistic regression over the 5 ratings (classification).

    p_worth = 1 - P(rating = 1);  e_gain = sum_k P(rating = k) * gain_k.
    """

    def __init__(self, builder: FeatureBuilder, C: float = 1.0, class_weight=None):
        self.builder, self.C, self.class_weight = builder, C, class_weight

    def fit(self, df):
        X = self.builder.fit_transform(df)
        self.m = LogisticRegression(C=self.C, max_iter=5000, class_weight=self.class_weight).fit(X, df["Label"].values)
        return self

    def predict(self, df):
        P = self.m.predict_proba(self.builder.transform(df))
        full = np.zeros((len(df), 5))
        full[:, self.m.classes_ - 1] = P
        return pd.DataFrame({"p_worth": 1 - full[:, 0], "e_gain": full @ GAINS}, index=df.index)


class BinaryRidge:
    """Binary logistic classifier for P(worthwhile) + ridge regression of gain."""

    def __init__(self, builder: FeatureBuilder, C: float = 1.0, alpha: float = 3.0):
        self.builder, self.C, self.alpha = builder, C, alpha

    def fit(self, df):
        X = self.builder.fit_transform(df)
        y = df["Label"].values
        self.clf = LogisticRegression(C=self.C, max_iter=5000).fit(X, (y > 1).astype(int))
        self.reg = Ridge(alpha=self.alpha).fit(X, gain(y))
        return self

    def predict(self, df):
        X = self.builder.transform(df)
        return pd.DataFrame(
            {"p_worth": self.clf.predict_proba(X)[:, 1], "e_gain": np.clip(self.reg.predict(X), 0, 1)}, index=df.index
        )


class GBM:
    """Histogram gradient-boosted trees: classifier + regressor on compact dense features."""

    def __init__(self, builder: FeatureBuilder, params: dict | None = None, seed: int = 0):
        self.builder, self.seed = builder, seed
        self.params = dict(max_iter=250, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0)
        self.params.update(params or {})

    def fit(self, df):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        X = self.builder.fit_transform(df).toarray()
        y = df["Label"].values
        self.clf = HistGradientBoostingClassifier(random_state=self.seed, **self.params).fit(X, (y > 1).astype(int))
        self.reg = HistGradientBoostingRegressor(random_state=self.seed, **self.params).fit(X, gain(y))
        return self

    def predict(self, df):
        X = self.builder.transform(df).toarray()
        return pd.DataFrame(
            {"p_worth": self.clf.predict_proba(X)[:, 1], "e_gain": np.clip(self.reg.predict(X), 0, 1)}, index=df.index
        )


class Ensemble:
    """Weighted average of member predictions (each column averaged separately)."""

    def __init__(self, members: list, weights: list[float] | None = None):
        self.members = members
        self.weights = np.ones(len(members)) if weights is None else np.asarray(weights, float)

    def fit(self, df):
        for m in self.members:
            m.fit(df)
        return self

    def predict(self, df):
        preds = [m.predict(df) for m in self.members]
        w = self.weights / self.weights.sum()
        out = sum(wi * p[["p_worth", "e_gain"]] for wi, p in zip(w, preds))
        for i, p in enumerate(preds):
            out[f"m{i}_gain"] = p["e_gain"].values
            out[f"m{i}_worth"] = p["p_worth"].values
        return out
