## Dataset Description

This dataset contains **3,090 research papers** for a personalized recommendation task. Your goal is to identify papers worth reading and prioritize stronger recommendations in the reading order.

Each paper includes bibliographic metadata. The training data also provides a rating reflecting one researcher’s reading preferences.

### Files

- **`train.csv`** — 2,494 papers with metadata and reading-preference ratings.
- **`test.csv`** — 596 candidate papers with metadata. Their ratings are hidden.
- **`sample_submission.csv`** — the required submission template containing all 596 test IDs.

### Columns

| **Column** | **Description** |
| :--------- | :-------------- |
| `id` | Unique paper identifier. |
| `title` | Paper title. |
| `venue` | Publication venue. |
| `year` | Publication year. |
| `authors` | Paper authors. |
| `doi` | Digital Object Identifier. |
| `Label` | Reading-preference rating from 1 to 5; available only in `train.csv`. |

### Reading-Preference Ratings

**Rating 1 means the researcher does not want to read the paper.** Ratings **2–5 indicate increasing reading priority**, with 5 representing the strongest preference.

These ratings reflect personal reading interests, rather than universal judgments of scientific quality.

### Submission Format

Submit a CSV with exactly two columns: **`id,priority`**. Include every test ID exactly once.

- **Priority 0:** do not recommend the paper.
- **Positive priority:** recommend the paper; higher values indicate earlier reading.

Priorities must be finite, nonnegative numbers. They do not need to use the 1–5 rating scale.

```csv
id,priority
1488,92.5
2524,0
2344,71.2
997,85.0
```

### Evaluation

The score combines **relevance** and **coverage** using their harmonic mean:

- **Relevance** rewards useful papers appearing earlier and penalizes recommendations with a true rating of 1.
- **Coverage** measures recall: the proportion of all worthwhile papers—those rated 2–5—that are recommended.

The test set is divided into **298 Public and 298 Private papers**. Membership is hidden, and recommendations are sorted and evaluated independently within each split. Final ranking uses the Private leaderboard.

See the **Evaluation** tab for the complete scoring formula.