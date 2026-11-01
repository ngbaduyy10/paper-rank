"""Pretrained sentence embeddings of titles, cached to disk.

Models are downloaded once from the Hugging Face Hub (public checkpoints) and
are used frozen; only titles (train + test, no labels) are encoded.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .data import ROOT

CACHE_DIR = ROOT / "cache"

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "specter2": "allenai/specter2_base",
    "bge": "BAAI/bge-base-en-v1.5",
    "bge_large": "BAAI/bge-large-en-v1.5",
    "e5_large": "intfloat/e5-large-v2",
    "mxbai": "mixedbread-ai/mxbai-embed-large-v1",
    "gte_large": "thenlper/gte-large",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
}
PREFIX = {"e5_large": "query: "}


def _build(name: str):
    from sentence_transformers import SentenceTransformer, models

    repo = MODELS[name]
    if name == "specter2":
        word = models.Transformer(repo, max_seq_length=64)
        pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="cls")
        return SentenceTransformer(modules=[word, pool], device="cpu")
    return SentenceTransformer(repo, device="cpu")


def embed_titles(titles: pd.Series, name: str = "minilm") -> np.ndarray:
    """L2-normalised embeddings for each title, keyed by title text in the cache."""
    titles = titles.fillna("").astype(str)
    key = hashlib.md5("\n".join(titles).encode()).hexdigest()[:10]
    path = CACHE_DIR / f"emb_{name}_{key}.npy"
    if path.exists():
        return np.load(path)
    model = _build(name)
    texts = [PREFIX.get(name, "") + t for t in titles]
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(path, emb)
    return emb


def all_titles(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    return pd.concat([train["title"], test["title"]], ignore_index=True)


def embedding_lookup(train: pd.DataFrame, test: pd.DataFrame, name: str) -> dict[int, np.ndarray]:
    """Map paper id -> embedding for every train and test paper."""
    titles = all_titles(train, test)
    emb = embed_titles(titles, name)
    ids = pd.concat([train["id"], test["id"]], ignore_index=True).values
    return dict(zip(ids, emb))
