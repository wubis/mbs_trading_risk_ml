# Commercial Mortgage Severity Analysis

This project answers two questions from the analyst assessment:

1. Do office properties show more distress than other property types?
2. How well can loan severity be estimated from the supplied loan and property data?

The analysis uses the required linear regression as a benchmark. It also tests a
two-stage hurdle model because severity is zero for most loans and continuous when a
loss occurs.

## Data

| File | Purpose | Rows |
| --- | --- | ---: |
| `loan_data.csv` | Labeled loans used for analysis and model development | 39,699 |
| `property_data.csv` | Property attributes joined on `propname` | 37,858 |
| `predictions.csv` | Unlabeled loans to score | 1,000 |
| `ml_assessment_analyst_-_v4_2026.docx` | Assessment instructions | N/A |

The property table has one row per `propname`. Both loan files join to it without
unmatched rows or changes in row count. The source CSV files are treated as read-only
inputs.

## Analysis process

### 1. Check the data contract

The script enforces the schemas, key uniqueness, merge relationship, target, and date
contracts. Source review also identified several issues that affect the analysis:

- `loan_id` is unique and is used only to identify output rows.
- `propname` identifies properties and is excluded from the models.
- 2,033 development properties back more than one loan.
- Severity is zero for 83.93% of labeled loans.
- 108 severity values exceed the stated upper bound of 1. The largest is 13.6473.
- The largest missing fields are square footage, occupancy, detailed property type,
  and year built.
- 848 rows imply a negative property age at origination.
- Six detailed property labels occur only in the scoring data.
- Training and scoring dates use different text formats but parse successfully.

The assignment's stated `[0, 1]` range is authoritative for the primary analysis, so
the 108 values above 1 are clipped to 1. The raw supplied target is retained as a
separately labeled linear-regression sensitivity.

### 2. Compare office and non-office loans

Part 1 assigns each loan to one of four property groups:

- `office`: CSSA property type is `OF`
- `mixed office`: any non-dedicated property whose detailed type contains `office`
- `non office`: no office exposure is identified
- `unknown mixed`: mixed-use property with missing detail

Four loans fall into the unknown group. They are reported separately and left out of
the requested three-group comparison.

| Property group | Loans | Mean severity | Positive severity rate | Mean severity when positive |
| --- | ---: | ---: | ---: | ---: |
| Office | 6,916 | 10.59% | 23.71% | 44.67% |
| Mixed-use with office | 1,088 | 7.93% | 18.47% | 42.92% |
| Non-office | 31,691 | 5.99% | 14.32% | 41.80% |

Office loans have the highest average severity. The conditional loss sizes are close
across the three groups, so most of the difference comes from office loans recording
a loss more often. This is an association in the supplied sample. It does not isolate
the effect of remote work or establish a causal relationship.

Two checks qualify the result. Loan-balance-weighted bounded mean severity is 7.88%
for office, 5.74% for mixed-use with office, and 7.00% for non-office. Using the raw
supplied values instead gives unweighted means of 10.64%, 7.95%, and 6.05%. Office
remains highest in both checks.

### 3. Define the prediction timing

The scoring file contains current occupancy, current NOI, and delinquency flags. The
primary model therefore treats the task as current-state severity estimation. These
fields would be target leakage if the intended task were prediction at origination.

An origination-only model is included as a timing sensitivity check. Its much weaker
result shows that the current-state fields provide most of the predictive signal.

### 4. Build the features

The model uses a compact set of fields with an economic interpretation:

- Loan size, LTV, origination occupancy, origination NOI, loan term, and vintage
- Property age, property size, broad property type, MSA category, state, and office
  group
- Current occupancy, occupancy change, current NOI, relative NOI change, and the two
  delinquency flags for the current-state models

Skewed size and income fields are transformed. Signed logarithms retain negative NOI
values. Invalid property ages are set to missing and paired with an invalid-age flag.
Numeric values are median-imputed with missingness indicators and standardized.
Categorical values are normalized, imputed, and one-hot encoded. The encoder ignores
previously unseen categories.

All imputation, scaling, and encoding are fitted on the training partition only.
Detailed property type is excluded because it contains hundreds of inconsistent
labels and several scoring-only categories.

### 5. Split by property

A random row split could place loans from the same property in both training and
validation. The model instead uses a fixed 80/20 `GroupShuffleSplit` on `propname`.

- Training set: 31,790 loans
- Validation set: 7,909 loans
- Property overlap: 0
- Positive severity: 16.16% in training and 15.72% in validation

The fixed holdout is used for the required plot and headline metrics. Model stability
is checked with three shuffled repetitions of five-fold `GroupKFold`. This gives 15
paired validation folds. Each property appears in one validation fold per repetition,
and preprocessing is refitted inside every fold.

A time-based test would be preferable for a future-cohort forecast. The data does not
include the date when severity was observed, so origination date alone cannot define
a clean chronological validation set.

### 6. Fit the required linear regression

Ordinary linear regression provides the requested baseline and a useful diagnostic.
It captures part of the signal but does not match the target distribution. Severity
has a large point mass at zero, a long right tail, and a nonnegative domain. Linear
regression predicts negative severity for 31.12% of validation loans.

### 7. Test a hurdle model

The hurdle model separates the problem into two parts:

1. A classifier estimates the probability that severity is greater than zero.
2. A regressor estimates severity conditional on a positive loss.

Expected severity is the product of those two estimates. This retains the continuous
target while handling the large number of zero observations. A classification-only
model would estimate loss occurrence but would not estimate loss size.

Both stages use histogram-based gradient-boosted trees. The conditional-magnitude
stage uses Poisson loss and clips magnitude estimates to `[0, 1]`, keeping final
expected-severity predictions inside the assignment's valid range.

## Validation results

| Model | R² | MSE | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| All-zero baseline | -0.122 | 0.042229 | 0.205498 | 0.067673 |
| Training-mean baseline | 0.000 | 0.037651 | 0.194038 | 0.117969 |
| Origination-only linear regression | 0.070 | 0.035026 | 0.187153 | 0.113832 |
| Current-state linear regression | 0.525 | 0.017876 | 0.133700 | 0.064777 |
| Current-state hurdle model | 0.596 | 0.015224 | 0.123384 | 0.047678 |

The hurdle model lowers MSE by about 15% relative to the current-state linear model
and by about 60% relative to the training-mean baseline. Its occurrence stage has a
ROC-AUC of 0.9355, average precision of 0.8064, and Brier score of 0.0548. Conditional
loss size remains the harder part of the problem, with MAE of 0.1871 among positive
validation cases.

Repeated grouped cross-validation produced the following fold averages:

| Model | Mean R² | Mean MSE | MSE standard deviation | Mean MAE |
| --- | ---: | ---: | ---: | ---: |
| Current-state linear regression | 0.525 | 0.017859 | 0.000629 | 0.064965 |
| Current-state hurdle model | 0.601 | 0.015015 | 0.000691 | 0.047549 |

The hurdle model had lower MSE and MAE in all 15 paired folds. It reduced mean MSE by
15.93% and mean MAE by 26.81%. Fold-level MSE improvement ranged from 13.35% to 18.16%.
The gain is moderate, but it is not driven by one favorable split.

The hurdle model has the lowest mean MSE in grouped cross-validation and is used to
score `predictions.csv`. The final file contains all 1,000 loans in source order with
no missing or infinite predictions. Scored severity ranges from 0.0001 to 0.7939.

## What the results support

The current model is a good assessment-level prototype. It improves clearly on the
required linear regression and produces estimates inside `[0, 1]`. The result does not
support a production claim yet. The largest open issue is whether current operating
and delinquency fields are valid at the intended prediction date.

The project stops at the hurdle model instead of running a large hyperparameter
search. The next useful work would be probability calibration, error review by
subgroup, and confirmation of the clipping policy with the data owner.

## Repository files

| Path | Contents |
| --- | --- |
| `part1.ipynb` | Part 1 checks, calculations, sensitivity analysis, and chart |
| `part1.txt` | Short written response for Part 1 |
| `part2.ipynb` | Executed Part 2 analysis, validation, plots, metrics, and scoring |
| `part2.py` | Reference script containing the same Part 2 modeling logic |
| `part2.txt` | Short written response for Part 2 |
| `WORKFLOW_DESIGN.md` | Implemented workflow, decisions, results, and limitations |
| `outputs/figures/` | Part 1 chart and Part 2 model-comparison plots |
| `outputs/metrics/` | Holdout metrics and grouped-CV summaries |
| `outputs/predictions_scored.csv` | Final scored copy of `predictions.csv` |

## Reproducing the work

The analysis was run with Python 3.12.7, pandas 2.2.3, NumPy 2.1.3,
scikit-learn 1.6.1, and Matplotlib 3.10.0. Jupyter is needed to run both notebooks.

From the repository root, restart each kernel and run all cells in `part1.ipynb` and
`part2.ipynb`. The Part 2 notebook recreates the figures, metric tables, and scored
predictions. The retained `part2.py` reference implementation can also be run from the
command line and accepts alternate paths through `python part2.py --help`. Both versions
use seed `20260901` and three repetitions of five-fold property-grouped cross-validation.

## Known limitations

- Repeated folds reuse observations across repetitions, so the fold spread is a
  stability check rather than 15 independent tests.
- There is no outcome observation date for a chronological test.
- Current-state variables may be leakage under an origination-time use case.
- The assessment clips severity above 1, but those source values still need a business
  definition before production use.
- Positive-loss magnitude is less accurate than loss-occurrence classification.
- Part 1 is descriptive and does not control for leverage, geography, vintage, or
  property quality.
