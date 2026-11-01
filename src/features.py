"""Feature builders. Every builder is fit on the training fold only, except the
unsupervised KMeans clustering, which may optionally see test titles (no labels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from .data import doi_type, split_authors


def gain(labels) -> np.ndarray:
    return (np.asarray(labels, dtype=float) - 1.0) / 4.0


class EmbeddingStore:
    """Holds frozen title embeddings for all train + test papers, indexed by id."""

    def __init__(self, ids, matrices: dict[str, np.ndarray]):
        self.pos = pd.Series(np.arange(len(ids)), index=np.asarray(ids))
        self.matrices = matrices

    def get(self, name: str, ids) -> np.ndarray:
        return self.matrices[name][self.pos.loc[np.asarray(ids)].values]

    def concat(self, names, ids) -> np.ndarray:
        return np.hstack([self.get(n, ids) for n in names])


def meta_frame(df: pd.DataFrame) -> pd.DataFrame:
    n_auth = df["authors"].map(lambda s: len(split_authors(s)))
    return pd.DataFrame(
        {
            "venue": df["venue"].values,
            "year": df["year"].astype(str).values,
            "doi_type": df["doi"].map(doi_type).values,
            "n_auth": np.minimum(n_auth, 6).astype(str).values,
        },
        index=df.index,
    )


class MetaOneHot:
    def __init__(self, cols=("venue", "year", "doi_type", "n_auth")):
        self.cols = list(cols)

    def fit(self, df):
        self.enc = OneHotEncoder(handle_unknown="ignore").fit(meta_frame(df)[self.cols])
        return self

    def transform(self, df):
        return self.enc.transform(meta_frame(df)[self.cols])


class TitleTfidf:
    """Word 1-2 gram + character 3-5 gram TF-IDF of the title."""

    def __init__(self, word=True, char=True):
        self.use_word, self.use_char = word, char

    def fit(self, df):
        t = df["title"]
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(t) if self.use_word else None
        self.char = (
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit(t)
            if self.use_char
            else None
        )
        return self

    def transform(self, df):
        parts = [v.transform(df["title"]) for v in (self.word, self.char) if v is not None]
        return sp.hstack(parts).tocsr()


class AuthorTargetEncoder:
    """Smoothed mean gain of each author's other papers (mean / max / count).

    Training rows are encoded out-of-fold so a paper never sees its own label.
    """

    def __init__(self, alpha: float = 3.0, n_splits: int = 5, seed: int = 0):
        self.alpha, self.n_splits, self.seed = alpha, n_splits, seed

    @staticmethod
    def _stats(df, g):
        rows = [(a, gi) for s, gi in zip(df["authors"], g) for a in split_authors(s)]
        if not rows:
            return pd.DataFrame(columns=["sum", "cnt"])
        t = pd.DataFrame(rows, columns=["a", "g"]).groupby("a")["g"].agg(["sum", "count"])
        return t.rename(columns={"count": "cnt"})

    def _encode(self, df, stats, prior):
        out = np.zeros((len(df), 3))
        for i, s in enumerate(df["authors"]):
            vals, cnts = [], []
            for a in split_authors(s):
                if a in stats.index:
                    su, c = stats.at[a, "sum"], stats.at[a, "cnt"]
                    vals.append((su + self.alpha * prior) / (c + self.alpha))
                    cnts.append(c)
            if vals:
                out[i] = [np.mean(vals) - prior, np.max(vals) - prior, np.log1p(np.sum(cnts))]
        return out

    def fit_transform(self, df):
        g = gain(df["Label"])
        self.prior = g.mean()
        self.stats = self._stats(df, g)
        out = np.zeros((len(df), 3))
        for a, b in KFold(self.n_splits, shuffle=True, random_state=self.seed).split(df):
            st = self._stats(df.iloc[a], g[a])
            out[b] = self._encode(df.iloc[b], st, g[a].mean())
        return out

    def transform(self, df):
        return self._encode(df, self.stats, self.prior)


class TopicClusters:
    """KMeans topic clusters on title embeddings (taught approach: clustering).

    Produces a one-hot cluster membership, and the smoothed mean gain / share of
    worthwhile papers of the cluster (computed from training-fold labels only).
    """

    def __init__(self, k: int = 30, alpha: float = 10.0, seed: int = 0):
        self.k, self.alpha, self.seed = k, alpha, seed

    def fit_kmeans(self, emb_all: np.ndarray):
        self.km = KMeans(self.k, n_init=10, random_state=self.seed).fit(emb_all)
        return self

    def fit(self, emb_train: np.ndarray, labels):
        c = self.km.predict(emb_train)
        g = gain(labels)
        w = (np.asarray(labels) > 1).astype(float)
        self.prior_g, self.prior_w = g.mean(), w.mean()
        cnt = np.bincount(c, minlength=self.k)
        self.c_gain = (np.bincount(c, g, self.k) + self.alpha * self.prior_g) / (cnt + self.alpha)
        self.c_worth = (np.bincount(c, w, self.k) + self.alpha * self.prior_w) / (cnt + self.alpha)
        return self

    def assign(self, emb):
        return self.km.predict(emb)

    def transform(self, emb, onehot=True):
        c = self.assign(emb)
        dense = np.column_stack([self.c_gain[c] - self.prior_g, self.c_worth[c] - self.prior_w])
        if not onehot:
            return dense
        oh = sp.csr_matrix((np.ones(len(c)), (np.arange(len(c)), c)), shape=(len(c), self.k))
        return sp.hstack([oh, sp.csr_matrix(dense)]).tocsr()


def knn_gain(emb_train: np.ndarray, labels, emb_query: np.ndarray, k: int = 25, exclude_self: bool = False):
    """Similarity-weighted mean gain and worthwhile share of the k nearest train titles."""
    sim = emb_query @ emb_train.T
    if exclude_self:
        np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    s = np.take_along_axis(sim, idx, axis=1)
    wts = np.exp(10 * s)
    g = gain(labels)[idx]
    wv = (np.asarray(labels)[idx] > 1).astype(float)
    return np.column_stack([(wts * g).sum(1) / wts.sum(1), (wts * wv).sum(1) / wts.sum(1), s.mean(1)])
