"""Loading and cleaning of the train / test paper metadata."""
from __future__ import annotations

import html
import io
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
COLUMNS = ["id", "title", "venue", "year", "authors", "doi"]

_RTF_HEX = re.compile(r"\\'([0-9a-fA-F]{2})")
_RTF_UC = re.compile(r"\\uc\d+ ?")
_RTF_UNICODE = re.compile(r"\\u(-?\d+) ?")


def decode_rtf_text(raw: str) -> str:
    """Extract the CSV body from a TextEdit RTF export and decode its escapes."""
    start = raw.find("id,title")
    if start < 0:
        raise ValueError("CSV header not found inside RTF file")
    body = raw[start:]
    body = body.replace("\\\n", "\n")
    body = re.sub(r"\}\s*$", "", body)
    body = _RTF_UC.sub("", body)
    body = _RTF_UNICODE.sub(lambda m: chr(int(m.group(1)) % 65536), body)
    body = _RTF_HEX.sub(lambda m: bytes([int(m.group(1), 16)]).decode("cp1252"), body)
    body = body.replace("\\{", "{").replace("\\}", "}").replace("\\\\", "\\")
    return body


def clean_text(s) -> str:
    if not isinstance(s, str):
        return ""
    s = html.unescape(html.unescape(s))
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["id"] = df["id"].astype(int)
    df["year"] = df["year"].astype(int)
    for col in ["title", "venue", "authors", "doi"]:
        df[col] = df[col].map(clean_text)
    df["venue"] = df["venue"].str.lower()
    return df


def load_train(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "train.csv", dtype=str, keep_default_na=False)
    df = _finalize(df)
    df["Label"] = df["Label"].astype(int)
    return df


def load_test(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "Test.csv"
    raw = Path(path).read_text(encoding="latin-1")
    text = decode_rtf_text(raw) if raw.lstrip().startswith("{\\rtf") else raw
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    return _finalize(df[COLUMNS])


def load_submission_ids(path: Path | None = None) -> list[int]:
    return pd.read_csv(path or DATA_DIR / "Submission.csv")["id"].astype(int).tolist()


def split_authors(s: str) -> list[str]:
    return [a.strip() for a in s.split(",") if a.strip()] if s else []


def doi_type(s: str) -> str:
    if s.startswith("10."):
        return "doi"
    if "semanticscholar" in s:
        return "s2"
    if "ceur-ws" in s:
        return "ceur"
    return "other"
