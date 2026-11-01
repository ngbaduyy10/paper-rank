# paper-rank: reading-priority recommendation

Assigns a reading priority to each of the 596 test papers. The aim is to bring highly rated papers forward while still recovering most of the papers worth reading. The final model combines four fine-tuned title encoders with a linear ensemble of multinomial logistic regression (classification), ridge regression (regression) and KMeans topic-cluster features (clustering). A cutoff that maximises the expected score decides which papers get priority 0.

The full write-up is in [docs/REPORT.md](docs/REPORT.md).

## Data

| File | Source | Notes |
| --- | --- | --- |
| `data/train.csv` | Kaggle competition | 2,494 papers, columns `id,title,venue,year,authors,doi,Label`. |
| `data/Test.csv` | Kaggle competition | 596 papers. The file is a TextEdit **RTF export** of a CSV, which `src/data.py::load_test` decodes, including `\'xx` and `\uN` escapes. |
| `data/Submission.csv` | Kaggle competition | Sample submission, used only for the list of test IDs. |

No additional labelled data is used. The pretrained checkpoints listed below are downloaded from the Hugging Face Hub. They are used either frozen, as title encoders, or as starting weights for fine-tuning on the training labels only.

| Role | Checkpoint |
| --- | --- |
| Frozen embeddings (final model) | `BAAI/bge-base-en-v1.5`, `allenai/specter2_base`, `sentence-transformers/all-mpnet-base-v2`, `sentence-transformers/all-MiniLM-L6-v2` |
| Frozen embeddings (experiments only) | `BAAI/bge-large-en-v1.5`, `intfloat/e5-large-v2`, `mixedbread-ai/mxbai-embed-large-v1`, `thenlper/gte-large`, `Qwen/Qwen3-Embedding-0.6B` |
| Fine-tuned (final model) | `BAAI/bge-base-en-v1.5`, `intfloat/e5-base-v2`, `sentence-transformers/all-mpnet-base-v2`, `allenai/scibert_scivocab_uncased` |

Preparation consists of HTML-entity unescaping, Unicode NFC normalisation, whitespace collapsing and lower-casing the venue (`src/data.py`). Embeddings are cached in `cache/emb_<model>_<hash>.npy`, keyed by the titles.

**Use of test metadata.** Test titles, but never labels, are encoded with the frozen models, and the KMeans topic clusters are fit on train and test titles together. Both steps are unsupervised. Every supervised step, including the cluster target statistics and author encodings, is fit on training rows only.

## Environment

- Python 3.9.6 and the pinned packages in `requirements.txt`.
- Hardware used: Apple M4 Pro with 24 GB RAM. Fine-tuning uses the PyTorch MPS backend when available and falls back to CPU otherwise.
- Seeds: `SEED = 42` for KMeans and the gradient-boosting experiment. CV folds use seed 0. Fine-tuning uses `torch.manual_seed(0)`, which also fixes the batch order.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If the matplotlib cache directory is not writable, set `MPLCONFIGDIR=$PWD/.mplcache`. Without it, the first plotting run can hang while building the font cache.

## Reproduce the submission

```bash
# 1. Fit the final model on all of train and write the priorities (about 11 min on M4 Pro / MPS)
.venv/bin/python scripts/make_submission.py
#    -> submissions/ens_final.csv       (id,priority)
#    -> submissions/ens_final_pred.csv  (P(worthwhile), expected gain per test paper)
```

The script checks that every test ID appears exactly once and that all priorities are finite and non-negative. The first run also downloads the checkpoints and computes the four embeddings.

The submitted file is preserved as `submissions/final.csv` (MD5 `e4a83ebcb718cdf9696ec72c1aa35ab9`, 381 of 596 papers recommended). Fine-tuning on MPS is not bit-for-bit deterministic, so a re-run can move a few papers across the cutoff and change priorities slightly. The linear part (`--model ens_main`) is fully deterministic. `submissions/ens_main.csv` is a lighter fallback with a CV score of 0.614.

## Local validation

```bash
# Out-of-fold predictions plus test-like simulation (writes results/oof/*.csv and results/cv_*.json)
.venv/bin/python scripts/validate.py --seeds 0 --models venue_prior tfidf_lr char_lr emb_lr emb_nometa_lr \
    emb_cluster_lr emb_author_lr full_lr emb_ridge tfidf_ridge gbm ens_main
.venv/bin/python scripts/validate.py --seeds 0 --models ft_bge ft_e5 ft_mpnet ft_scibert   # about 5 min each
.venv/bin/python scripts/validate.py --seeds 0 --models ens_ft4 ens_final                  # composed from member OOFs

.venv/bin/python scripts/eda.py       # results/eda.json, results/figures/eda_overview.png
.venv/bin/python scripts/analyze.py   # results/ablation.csv, error analysis, figures
```

The protocol is as follows. A 5-fold `StratifiedGroupKFold` is stratified by venue × label and grouped by normalised title, so duplicate titles never cross folds. This gives out-of-fold predictions for every training paper. The simulation then repeatedly draws 596-paper pseudo test sets with the real test venue mix (cav 221, lics 187, kr 151, lpnmr 37, no iclp) and applies the cutoff strategy to the whole set, as for the real file. Each set is split at random into two 298-paper halves, and each half is scored with the exact competition metric (`src/metric.py`).

Expected results (seed 0, 200 pseudo test sets, so 400 scored splits):

| Model | AUC P(rating>1) | Score (all recommended) | Score (adaptive cutoff) | Q | R |
| --- | --- | --- | --- | --- | --- |
| `venue_prior` | 0.612 | 0.522 | 0.519 | 0.365 | 0.897 |
| `ens_main` (linear ensemble) | 0.783 | 0.562 | 0.614 | 0.487 | 0.830 |
| `ens_final` (submitted) | 0.815 | 0.563 | **0.626 ± 0.020** | 0.510 | 0.812 |

For reference, recommending every paper in random order scores 0.484. An oracle that excludes all rating-1 papers and orders the rest perfectly scores 0.757.

## Layout

```
src/data.py         loaders (train CSV, RTF test), text cleaning
src/features.py     metadata one-hot, TF-IDF, author target encoding, KMeans clusters, kNN features
src/embeddings.py   frozen sentence-transformer title embeddings (cached)
src/finetune.py     fine-tuned encoder with 5-class rating head
src/models.py       OrdinalLR (classification), BinaryRidge (classification + regression), GBM, Ensemble
src/cutoff.py       priorities and the priority-0 cutoff (expected-score maximisation)
src/metric.py       exact competition metric
src/validation.py   CV splits and test-mix simulation
src/pipeline.py     named model configurations, FINAL_MODEL / FINAL_STRATEGY
scripts/            validate.py, make_submission.py, eda.py, analyze.py
results/            CV JSONs, OOF predictions, ablation table, error analysis, figures
```

## Limitations

- Validation can only mimic the test venue mix. It cannot detect topic drift within a venue between train and test.
- Fine-tuning on MPS is not fully deterministic (see above).
- Only one CV repeat (seed 0) was used for the fine-tuned models. Model gaps under about 0.005 are within noise.
