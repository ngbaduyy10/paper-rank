## 1. Task and teams

Build a system that assigns a reading priority to every paper in the test collection. The reading order should bring relevant papers forward and provide broad title coverage as reading continues.

- A team may have **up to five members**. A student may belong to only one team.
- State the Kaggle team name, each member's name and student ID, and individual contributions in the report.
- Submit one finite numeric `priority` for each of the **596 test IDs**, using the columns `id,priority`. Each ID must appear exactly once.
- Higher priorities indicate earlier reading. The numeric scale is arbitrary; the evaluator uses the resulting order, without converting priorities to predicted 1–5 labels.
- Follow the submission limits and deadlines announced on Kaggle and the LMS. Select the final submission(s) permitted by the competition before its deadline.

## 2. Required approaches

Your system must use **at least one** of the following approaches taught in class as a substantive foundation:

1. Classification.
2. Regression.
3. Clustering.
4. Itemset-based recommendation, using frequent itemsets or association rules.

You do **not** need to implement all four approaches. You may combine them and use additional techniques beyond the course scope. Explain how the taught approach contributes to the final priorities and how any additional techniques support the solution. Merely mentioning an approach without using it in the implemented system does not satisfy this requirement.

## 3. Assignment grade

All component grades are on a **0–10 scale**. The final assignment grade is:

$$
\boxed{A = 0.60L + 0.40R}
$$

- **L:** ranking grade, converted from the final Private leaderboard position.
- **R:** report grade, awarded using the rubric below.

For example, a ranking grade of 9.5 and a report grade of 8.0 produce an assignment grade of `0.60 × 9.5 + 0.40 × 8.0 = 8.9` -> Round to 9 (Round to the nearest on scale 0.5).

These are course grading weights. The competition's raw score separately combines relevance and coverage, as specified on the Evaluation page. The raw Kaggle score is not directly multiplied by 10 to obtain L.

## 4. Converting leaderboard rank to L

Use the **final Private leaderboard** and count only eligible course teams. Public leaderboard results provide feedback during the competition and do not determine the ranking grade.

| **Position among eligible teams** | **Ranking grade** |
| :--- | :--- |
| Top 10% | 10.0 |
| Above 10% through 20% | 9.5 |
| Above 20% through 30% | 9.0 |
| Above 30% through 40% | 8.75 |
| Above 40% through 50% | 8.5 |
| Above 50% through 60% | 8.25 |
| Above 60% through 70% | 8 |
| Above 70% through 80% | 7.75 |
| Above 80% through 90% | 7.5 |
| Above 90% through 100% | 7.0 |
| If the Public score lower than Public score of Baseline: 6.5 | |

For N eligible teams, each cumulative cutoff is rounded upward: top 10% includes the first `ceil(0.10 × N)` teams, top 20% includes the first `ceil(0.20 × N)` teams, and so on. Use the first band whose cumulative cutoff includes the team's rank. Teams with exactly equal underlying Private scores receive the same ranking grade, using the best position occupied by their tied group. Equality of rounded display values alone does not establish a tie.

Non-submissions, invalid submissions, and integrity violations are handled under the course assessment policy and are not covered by this floor.

## 5. Required deliverables and reproducibility

Submit all of the following by the announced deadlines:

1. A valid Kaggle submission and the final selected submission(s).
2. A GitHub repository containing the code used to generate the submission. A private repository is acceptable if the teaching team (tuvn@hcmut.edu.vn) has access.
3. A report containing the GitHub URL, final commit hash, Kaggle team name, and submission identifier(s). Preserve the exact submitted CSV.

The repository README must include:

- The input files and any additional data, with sources and preparation steps.
- Environment setup, dependency versions, and random seeds.
- The notebook or exact commands for training and generating `id,priority`.
- The local validation procedure, expected validation results, and instructions for reproducing the selected submission.
- Relevant hardware, estimated runtime, and any limitations affecting reproduction.

The report must connect **domain understanding → data analysis → insights → solution choices → implementation → evaluation → conclusions**. Explain why each major choice addresses the identified needs or findings.

Students reproduce the submitted priorities from the available data; they do not receive hidden test labels. The teaching team uses the hidden answer key and official metric to verify the resulting competition score. Results that cannot be reproduced cannot receive full credit for implementation and evaluation evidence.

## 6. Report rubric: R out of 10

| **Criterion** | **Maximum points** | **Evidence expected for full credit** |
| :--- | :--- | :--- |
| **1. Presentation and clarity** | **1.0** | A coherent structure, precise terminology, readable figures and tables, appropriate citations, and clear explanations that allow a reader to follow the work. |
| **2. Problem formulation and domain understanding** | **1.5** | Identify the researcher or stakeholder, their reading needs, the required inputs and outputs, constraints, and the meaning of relevance and coverage. Explain how the metric relates to the practical objective. |
| **3. Data analysis** | **1.5** | Examine distributions, missing values, duplicates, label imbalance, metadata and text characteristics, and relationships relevant to the task. Explain cleaning and representation decisions using evidence. |
| **4. Insights from domain and data analysis** | **1.5** | Derive specific, supported findings from the domain process and data. Distinguish observations from assumptions; explain their implications and how they inform testable expectations or design choices. |
| **5. Solution design and implementation** | **2.0** | Justify the chosen approach, connect it to the insights, and explain how it produces priorities. Describe preprocessing, representation, algorithms, and important settings. Show a substantive taught approach and an implementation reproducible from GitHub. |
| **6. Method evaluation, error analysis, and fairness** | **2.0** | Use a justified validation design and the competition metric. Compare with a baseline under comparable conditions; analyze relevance and coverage, ranking failures, and representative error cases. Address leakage, sampling bias, leaderboard overfitting, and variability or uncertainty in the results. |
| **7. Conclusions and general insights** | **0.5** | Summarize what the evidence supports, connect findings to the original need, state limitations, and discuss what may generalize and what should be investigated next. |

Analysis requires interpretation and supporting evidence. Lists of tools, unexplained plots, and leaderboard screenshots alone do not establish the requested insights or justify the solution.

## 7. Fair evaluation and academic integrity

- Train set can be used for both evaluation and training. How to split, handled them based on student's choice.
- Disclose any unsupervised use of visible test metadata and use a consistent, defensible protocol in local validation.
- Compare methods using the same validation partitions and scoring rules. Explain how splits handle duplicates or related papers where relevant.
- Report component scores and representative failures, not only the most favorable total score. **Do not treat repeated Public leaderboard tuning as independent evidence of generalization.**- Publicly accessible external data, pretrained models, and third-party code may be used if identified, cited, and explained. Teams must understand and justify their submitted work.
- General discussion is allowed. **Exchanging another team's implementation, submission files, or complete solution is not allowed.**
- **MUST NOT attempt to obtain hidden test labels, manually construct a test answer key, recover the hidden Public/Private membership, or exploit feedback to infer individual hidden labels.**

## 8. Announcement of grades

**Report grades and final assignment grades will be announced approximately two weeks after the assignment ends.** This period allows the teaching team to review reports, verify reproducibility, and check submitted work.