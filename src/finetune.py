"""Optional: fine-tune a small pretrained encoder as a rating regressor/classifier.

Input text = "<venue> <year>: <title>". Head predicts the 5-class rating
distribution (cross-entropy), giving p_worth and e_gain like the other models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAINS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def _texts(df):
    return (df["venue"] + " " + df["year"].astype(str) + ": " + df["title"]).tolist()


class FineTuned:
    def __init__(self, repo="BAAI/bge-base-en-v1.5", epochs=4, lr=3e-5, batch_size=32, max_len=48, seed=0):
        self.repo, self.epochs, self.lr, self.bs, self.max_len, self.seed = repo, epochs, lr, batch_size, max_len, seed

    def fit(self, df):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        torch.manual_seed(self.seed)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.repo)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.repo, num_labels=5).to(self.device)
        enc = self.tok(_texts(df), padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        y = torch.tensor(df["Label"].values - 1)
        n = len(y)
        steps = self.epochs * int(np.ceil(n / self.bs))
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        g = torch.Generator().manual_seed(self.seed)
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.bs):
                b = perm[i : i + self.bs]
                out = self.model(
                    input_ids=enc["input_ids"][b].to(self.device),
                    attention_mask=enc["attention_mask"][b].to(self.device),
                    labels=y[b].to(self.device),
                )
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
        return self

    def predict(self, df):
        import torch

        self.model.eval()
        enc = self.tok(_texts(df), padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        probs = []
        with torch.no_grad():
            for i in range(0, len(df), 128):
                logits = self.model(
                    input_ids=enc["input_ids"][i : i + 128].to(self.device),
                    attention_mask=enc["attention_mask"][i : i + 128].to(self.device),
                ).logits
                probs.append(torch.softmax(logits, -1).cpu().numpy())
        P = np.vstack(probs)
        return pd.DataFrame({"p_worth": 1 - P[:, 0], "e_gain": P @ GAINS}, index=df.index)
