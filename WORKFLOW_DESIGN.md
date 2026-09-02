# MBS Severity Analysis: Implementation Record

**Status:** Complete and reproducible

**Last verified:** 2026-09-01

**Scope:** Document the implemented workflow, modeling decisions, generated artifacts, and remaining limitations for the two-part analyst assessment.

## 1. Delivered assessment artifacts

The analysis is complete around the four files required by the assessment:

| File | Implemented contents |
| --- | --- |
| `part1.ipynb` | Merge validation, office classification, severity summaries, sensitivity checks, and the Part 1 figure |
| `part1.txt` | Concise interpretation of the office-distress comparison |
| `part2.ipynb` | Executed data validation, feature engineering, grouped evaluation, required OLS baseline, hurdle-model experiment, plots, and scoring |
| `part2.txt` | Feature rationales, required metrics, OLS diagnosis, and alternative-model recommendation |

Supporting repository artifacts include `README.md`, generated figures and metric tables under `outputs/`, and `outputs/predictions_scored.csv`. The supplied CSV files are treated as read-only inputs.

For final delivery, the safest interpretation of the assessment instructions is to submit only the four required files above. The source CSVs and supporting repository artifacts remain available for reproducibility.

## 2. Data contract and validated relationships

| Input | Role | Validated state |
| --- | --- | ---: |
| `loan_data.csv` | Labeled development loans | 39,699 rows x 13 columns |
| `property_data.csv` | Property attributes | 37,858 rows x 8 columns |
| `predictions.csv` | Unlabeled scoring loans | 1,000 rows x 13 columns; severity is blank |
| `ml_assessment_analyst_-_v4_2026.docx` | Assessment specification | 3 pages |

The implemented loaders enforce the following contracts:

- `loan_id` is unique in both loan files and is excluded from modeling.
- `propname` is unique in `property_data.csv` and defines a many-to-one loan-to-property merge.
- Every development and scoring loan matches exactly one property row without changing row counts.
- Training severity is populated and nonnegative; scoring severity is entirely blank.
- `is_90d` implies `is_30d` in the supplied data.
- Both source date formats parse successfully.
- The scoring output preserves all 1,000 loan IDs in source order.

There are 36,978 development properties, including 2,033 that back multiple loans. There are also 116 scoring properties present in the development sample. Identifiers are not used as features, and validation is grouped by `propname` to prevent property overlap between training and validation.

## 3. Target and data-quality decisions

Severity has two properties that drive the modeling design:

- 33,319 of 39,699 observations (83.93%) are exactly zero.
- 108 observations exceed the specification's stated upper bound of one; the maximum is 13.6473.

The specification's stated `[0, 1]` range is authoritative for the primary analyses, so values above one are clipped to one. The raw supplied target is retained as a separately labeled linear-model sensitivity so the impact of that policy remains visible.

Other handled issues include:

| Issue | Implemented treatment |
| --- | --- |
| Missing occupancy and square footage | Training-fold median imputation with missingness indicators |
| Negative current NOI | Signed-log transformation |
| 848 negative implied property ages | Invalid-age flag, then missing-value imputation |
| Different train/scoring date formats | Explicit date parsing before feature construction |
| High-cardinality detailed property labels | Excluded from the baseline; broad normalized categories are used instead |
| Scoring-only category labels | One-hot encoding uses `handle_unknown="ignore"` |

## 4. Part 1 implementation

The unit of analysis is one loan because severity is supplied at loan level. Property attributes are joined with:

```python
loans.merge(properties, on="propname", how="left", validate="many_to_one")
```

Office exposure is assigned deterministically:

1. `Office`: `cssaproptype == "OF"`.
2. `Mixed-use with office`: any non-dedicated property whose detailed type contains the word `office`, even when its broad CSSA code is not `MU`.
3. `Non-office`: no identified office exposure.
4. `Unknown mixed-use`: mixed-use property with missing detail.

Four unknown mixed-use loans are disclosed and excluded from the requested three-group comparison.

| Property group | Loans | Mean severity | Positive-severity rate | Mean severity if positive |
| --- | ---: | ---: | ---: | ---: |
| Office | 6,916 | 10.59% | 23.71% | 44.67% |
| Mixed-use with office | 1,088 | 7.93% | 18.47% | 42.92% |
| Non-office | 31,691 | 5.99% | 14.32% | 41.80% |

Office loans have the highest unweighted mean severity. Most of the difference comes from a higher frequency of positive severity rather than a substantially larger loss conditional on loss.

Two sensitivity checks preserve the conclusion:

- Loan-balance-weighted bounded mean severity is 7.88% for office, 5.74% for mixed-office, and 7.00% for non-office loans.
- With raw supplied severity, the unweighted means are 10.64%, 7.95%, and 6.05%, respectively.

The comparison is descriptive. It does not isolate remote work as a cause or control for leverage, delinquency, geography, vintage, property quality, and other confounders.

## 5. Part 2 feature and validation design

The primary task is interpreted as current-state severity estimation because the scoring file contains current occupancy, current NOI, and delinquency flags. An origination-only linear model quantifies sensitivity to the alternative interpretation in which those fields would be unavailable.

The implemented feature set includes:

- Loan size, LTV, origination occupancy, origination NOI, loan term, and vintage.
- Property age, invalid-age flag, property size, broad property type, MSA category, state, and office group.
- Current occupancy, occupancy change, current NOI, relative NOI change, and delinquency flags for current-state models.

Skewed size and income fields are transformed. Numeric features are median-imputed with missingness indicators and standardized. Categorical features are normalized, imputed, and one-hot encoded. All fitted preprocessing occurs inside the training partition or cross-validation fold.

The fixed holdout uses an 80/20 `GroupShuffleSplit` on `propname`:

- Training: 31,790 loans.
- Validation: 7,909 loans.
- Property overlap: zero.
- Positive severity: 16.16% in training and 15.72% in validation.

Stability is checked with three shuffled repetitions of five-fold `GroupKFold`, producing 15 paired model comparisons. Preprocessing and estimation are refitted in every fold.

## 6. Implemented models and results

Ordinary least squares is the required baseline. It is compared with constant baselines, an origination-only OLS sensitivity model, and an optional hurdle model.

The hurdle model has two stages:

1. A histogram gradient-boosted classifier estimates whether severity is positive.
2. A histogram gradient-boosted Poisson regressor estimates severity conditional on a positive loss.

Expected severity is the product of the two stage predictions. The conditional magnitude is clipped to `[0, 1]`, which keeps final predictions inside the stated target range.

| Model | Holdout R-squared | Holdout MSE | Holdout RMSE | Holdout MAE |
| --- | ---: | ---: | ---: | ---: |
| All-zero baseline | -0.122 | 0.042229 | 0.205498 | 0.067673 |
| Training-mean baseline | 0.000 | 0.037651 | 0.194038 | 0.117969 |
| Origination-only linear regression | 0.070 | 0.035026 | 0.187153 | 0.113832 |
| Current-state linear regression | 0.525 | 0.017876 | 0.133700 | 0.064777 |
| Current-state hurdle model | 0.596 | 0.015224 | 0.123384 | 0.047678 |

The required linear model predicts negative severity for 31.12% of validation loans. The hurdle model produces predictions inside `[0, 1]` and lowers holdout MSE by approximately 15%.

Across the 15 paired grouped folds:

| Model | Mean R-squared | Mean MSE | MSE standard deviation | Mean MAE |
| --- | ---: | ---: | ---: | ---: |
| Current-state linear regression | 0.525 | 0.017859 | 0.000629 | 0.064965 |
| Current-state hurdle model | 0.601 | 0.015015 | 0.000691 | 0.047549 |

The hurdle model has lower MSE and MAE in all 15 folds. Mean MSE falls by 15.93%, and mean MAE falls by 26.81%.

## 7. Generated outputs

Running `part1.ipynb` and `part2.ipynb` creates or refreshes:

```text
outputs/
├── figures/
│   ├── part1_office_severity.png
│   ├── part2_linear_actual_vs_predicted.png
│   └── part2_hurdle_actual_vs_predicted.png
├── metrics/
│   ├── part2_metrics.csv
│   ├── part2_repeated_group_cv_comparison.csv
│   └── part2_repeated_group_cv_summary.csv
└── predictions_scored.csv
```

The model with the lowest mean grouped-CV MSE is refitted on all labeled rows before scoring. The hurdle model is selected under this rule. The final output contains 1,000 finite predictions in source order, ranging from 0.0001 to 0.7939, and does not overwrite `predictions.csv`.

## 8. Reproduction and verification

The recorded environment is Python 3.12.7 with pandas 2.2.3, NumPy 2.1.3, scikit-learn 1.6.1, and Matplotlib 3.10.0. Jupyter is required for both parts.

From the repository root:

1. Restart the notebook kernel and run all cells in `part1.ipynb`.
2. Restart the kernel and run all cells in `part2.ipynb`.
3. Confirm the written metrics match the generated figures and CSV tables.
4. Confirm Git shows no changes to the three supplied input CSVs.

The current checked-in artifacts have been reproduced from a clean execution after the office-classification and bounded-target updates. The written results, metric tables, figures, and scoring output reflect that same run.

## 9. Known limitations and next work

- Current occupancy, NOI, and delinquency are valid only for a current-state use case; they would leak information into an origination-time forecast.
- There is no outcome observation date for a clean chronological validation split.
- The assessment clips severity above one, but those source values require a business definition before production use.
- The hurdle model's conditional loss magnitude remains less accurate than its loss-occurrence stage.
- Part 1 is descriptive and does not adjust for confounding variables.
- The repository records package versions in documentation but does not include a dependency lock file or a separate automated test suite.
- Model selection and final performance estimation would be separated more strictly in a production study, ideally with nested grouped validation or an untouched final holdout.

The assessment deliverables are complete. Further work should focus on timing semantics, probability calibration, subgroup error analysis, confirming the target policy with the data owner, and production validation rather than adding more model complexity.
