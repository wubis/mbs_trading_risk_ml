#!/usr/bin/env python3
"""Part 2: model commercial-mortgage severity and score held-out loans.

The script performs modeling-readiness EDA, validates the data contract, builds
leakage-aware feature sets, evaluates constant and linear-regression baselines
on a property-grouped holdout, experiments with a two-stage hurdle model, and
scores predictions.csv without modifying the supplied input files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 20260901
VALIDATION_SIZE = 0.20

EXPECTED_LOAN_COLUMNS = [
    "loan_id",
    "propname",
    "orig_date",
    "end_date",
    "loan_size_mm",
    "ltv",
    "occ_at_orig",
    "occ",
    "noi_at_orig",
    "noi",
    "is_30d",
    "is_90d",
    "severity",
]
EXPECTED_PROPERTY_COLUMNS = [
    "propname",
    "cssaproptype",
    "proptype",
    "proptypelong",
    "state",
    "msa_category",
    "year_built",
    "sqft",
]

IDENTIFIER_COLUMNS = ["loan_id", "propname"]
TEMPORAL_COLUMNS = ["orig_date", "end_date"]
TARGET_COLUMN = "severity"
ORIGINATION_NUMERIC_FEATURES = [
    "log_loan_size",
    "ltv_fraction",
    "occ_at_orig_fraction",
    "signed_log_noi_orig",
    "loan_term_years",
    "origination_year",
    "property_age",
    "property_age_invalid",
    "log_sqft",
]
CURRENT_STATE_NUMERIC_FEATURES = ORIGINATION_NUMERIC_FEATURES + [
    "occ_fraction",
    "occupancy_change",
    "signed_log_noi_current",
    "noi_relative_change",
    "is_30d",
    "is_90d",
]
CATEGORICAL_FEATURES = [
    "cssaproptype",
    "msa_category",
    "state",
    "office_group",
]

# These rationales are printed by the script and copied into the written response.
# Keeping them beside the feature lists prevents the documentation from drifting away
# from the actual model specification.
FEATURE_RATIONALES = {
    "log_loan_size": (
        "Loan balance measures exposure size; log1p reduces the influence of very "
        "large loans."
    ),
    "ltv_fraction": (
        "Higher leverage leaves less collateral cushion if a property must be sold."
    ),
    "occ_at_orig_fraction": (
        "Origination occupancy records the property's starting demand and "
        "underwriting condition."
    ),
    "signed_log_noi_orig": (
        "Origination NOI measures starting debt-service capacity; the signed log "
        "limits scale skew."
    ),
    "loan_term_years": (
        "Contractual loan term may change refinancing and maturity risk."
    ),
    "origination_year": (
        "Origination vintage can proxy for underwriting and market regimes."
    ),
    "property_age": ("Property age can capture physical and functional obsolescence."),
    "property_age_invalid": (
        "An indicator preserves information that the recorded building year is "
        "inconsistent with origination."
    ),
    "log_sqft": (
        "Property size can affect liquidity and recovery; log1p reduces its heavy "
        "right skew."
    ),
    "occ_fraction": (
        "Current occupancy measures the property's latest operating health under "
        "the current-state prediction assumption."
    ),
    "occupancy_change": (
        "The change from origination measures deterioration or improvement rather "
        "than only the current level."
    ),
    "signed_log_noi_current": (
        "Current NOI measures latest income support while retaining economically "
        "meaningful negative values."
    ),
    "noi_relative_change": (
        "Relative NOI change captures operating deterioration with a stable "
        "denominator and fixed tail clipping."
    ),
    "is_30d": (
        "Thirty-day delinquency is an observed indicator of current credit stress."
    ),
    "is_90d": "Ninety-day delinquency identifies more advanced payment distress.",
    "cssaproptype": (
        "The normalized broad property type captures sector-specific risk without "
        "high-cardinality detail labels."
    ),
    "msa_category": (
        "MSA category provides a stable, low-cardinality proxy for market depth "
        "and location."
    ),
    "state": (
        "State captures additional geographic variation and is encoded with "
        "unseen-category protection."
    ),
    "office_group": (
        "The Part 1 office-exposure grouping represents the observed difference "
        "between office, mixed-office, and non-office loans."
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line paths and the reproducibility seed.

    Returns:
        Parsed command-line arguments. File defaults are resolved relative to this
        script so execution does not depend on the caller's working directory.
    """
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loan-data", type=Path, default=root / "loan_data.csv")
    parser.add_argument(
        "--property-data", type=Path, default=root / "property_data.csv"
    )
    parser.add_argument(
        "--prediction-data", type=Path, default=root / "predictions.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def print_section(title: str) -> None:
    """Print a consistently formatted console section heading.

    Args:
        title: Human-readable section title.
    """
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def require(condition: bool, message: str) -> None:
    """Raise a data-contract error when a required condition is false.

    Unlike ``assert``, this check remains active when Python is run with the ``-O``
    optimization flag. It is therefore appropriate for input and output validation.

    Args:
        condition: Condition that must evaluate to true.
        message: Error message explaining the violated contract.

    Raises:
        ValueError: If ``condition`` is false.
    """
    if not condition:
        raise ValueError(message)


def normalized_text(series: pd.Series) -> pd.Series:
    """Normalize categorical labels while preserving missing values.

    Args:
        series: Raw string-like categorical values.

    Returns:
        Pandas string series with trimmed, case-folded labels and normalized word
        separators. Missing inputs remain missing rather than becoming the text
        ``"nan"``.
    """
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .str.replace(r"[\s_-]+", " ", regex=True)
    )


def load_and_validate(
    loan_path: Path, property_path: Path, prediction_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, validate, and merge the development and scoring inputs.

    Args:
        loan_path: Labeled loan-level development CSV.
        property_path: Property-level attribute CSV.
        prediction_path: Unlabeled loan-level scoring CSV.

    Returns:
        A tuple containing raw loans, raw properties, merged development rows, and
        merged scoring rows.

    Raises:
        ValueError: If a schema, key, target, delinquency, row-count, or join
        contract is violated.
        pandas.errors.MergeError: If the property merge is not many-to-one.
    """
    loans = pd.read_csv(loan_path)
    properties = pd.read_csv(property_path)
    scoring = pd.read_csv(prediction_path)

    # Validate schemas before accessing named fields so failures are direct and useful.
    require(loans.columns.tolist() == EXPECTED_LOAN_COLUMNS, "Unexpected loan schema")
    require(
        scoring.columns.tolist() == EXPECTED_LOAN_COLUMNS,
        "Unexpected scoring schema",
    )
    require(
        properties.columns.tolist() == EXPECTED_PROPERTY_COLUMNS,
        "Unexpected property schema",
    )

    # IDs define row identity, while propname defines the many-to-one join contract.
    require(loans["loan_id"].is_unique, "Training loan_id must be unique")
    require(scoring["loan_id"].is_unique, "Scoring loan_id must be unique")
    require(
        set(loans["loan_id"]).isdisjoint(scoring["loan_id"]),
        "Train/scoring loan IDs overlap",
    )
    require(
        properties["propname"].is_unique,
        "Property-side propname must be unique",
    )

    # Development rows must carry labels; scoring labels must remain entirely blank.
    require(
        loans[TARGET_COLUMN].notna().all(),
        "Training severity must be populated",
    )
    require(
        scoring[TARGET_COLUMN].isna().all(),
        "Scoring severity must be empty",
    )
    require(loans[TARGET_COLUMN].ge(0).all(), "Severity must be nonnegative")
    require(
        (~loans["is_90d"] | loans["is_30d"]).all(),
        "is_90d must imply is_30d",
    )

    development = loans.merge(
        properties,
        on="propname",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    score_merged = scoring.merge(
        properties,
        on="propname",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    # A left join must preserve every loan and match it to exactly one property row.
    require(len(development) == len(loans), "Development merge changed row count")
    require(len(score_merged) == len(scoring), "Scoring merge changed row count")
    require(
        development["_merge"].eq("both").all(),
        "Unmatched development properties",
    )
    require(
        score_merged["_merge"].eq("both").all(),
        "Unmatched scoring properties",
    )

    development = development.drop(columns="_merge")
    score_merged = score_merged.drop(columns="_merge")
    return loans, properties, development, score_merged


def make_feature_audit(
    development: pd.DataFrame, scoring: pd.DataFrame
) -> pd.DataFrame:
    """Summarize feature roles, timing, missingness, and category coverage.

    Args:
        development: Merged labeled development rows.
        scoring: Merged unlabeled scoring rows.

    Returns:
        One audit row per raw column, including model role, availability timing,
        missing counts, cardinality, and unseen scoring-category count.
    """
    # Detailed property labels are audited even though the baseline deliberately
    # excludes them; unseen categories are an important train/score compatibility risk.
    categorical = {
        "cssaproptype",
        "proptype",
        "proptypelong",
        "state",
        "msa_category",
    }
    identifiers = set(IDENTIFIER_COLUMNS)
    temporal = set(TEMPORAL_COLUMNS)
    current_state = {"occ", "noi", "is_30d", "is_90d"}
    rows = []
    for column in development.columns:
        # Assign each raw field one mutually exclusive modeling role.
        if column == TARGET_COLUMN:
            role = "target"
        elif column in identifiers:
            role = "identifier / excluded"
        elif column in temporal:
            role = "temporal / derive"
        elif column in categorical:
            role = "categorical"
        elif pd.api.types.is_bool_dtype(development[column]):
            role = "boolean"
        else:
            role = "numeric"

        # Timing is tracked separately because a numeric field can still leak future
        # information if it was not known at the intended prediction date.
        if column in identifiers:
            timing = "identifier"
        elif column == TARGET_COLUMN:
            timing = "outcome"
        elif column in current_state:
            timing = "current state"
        else:
            timing = "known at origination"

        # An unseen scoring category must not crash the fitted one-hot encoder.
        unseen_categories = 0
        if column in categorical:
            train_values = set(development[column].dropna().astype(str))
            score_values = set(scoring[column].dropna().astype(str))
            unseen_categories = len(score_values - train_values)

        rows.append(
            {
                "column": column,
                "role": role,
                "timing": timing,
                "dtype": str(development[column].dtype),
                "train_missing": int(development[column].isna().sum()),
                "score_missing": int(scoring[column].isna().sum()),
                "train_unique": int(development[column].nunique(dropna=True)),
                "unseen_score_categories": unseen_categories,
            }
        )
    return pd.DataFrame(rows)


def classify_office_group(frame: pd.DataFrame) -> np.ndarray:
    """Apply the Part 1 office-exposure classification.

    Args:
        frame: Rows containing CSSA, broad, and detailed property-type fields.

    Returns:
        Array with one of ``office``, ``mixed office``, ``unknown mixed``, or
        ``non office`` for each input row.
    """
    cssa = normalized_text(frame["cssaproptype"])
    broad_type = normalized_text(frame["proptype"])
    detailed_type = normalized_text(frame["proptypelong"])
    is_office = cssa.eq("of")
    is_mixed = cssa.eq("mu") | broad_type.eq("mixed use")
    has_office = detailed_type.str.contains(r"\boffice\b", regex=True, na=False)
    unknown_mixed = is_mixed & detailed_type.isna()
    # Classification precedence matters: dedicated office properties must not be
    # absorbed into a generic mixed-use or default category.
    return np.select(
        [is_office, is_mixed & has_office, unknown_mixed],
        ["office", "mixed office", "unknown mixed"],
        default="non office",
    )


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the shared, leakage-aware feature representation.

    Args:
        frame: Merged raw loan and property rows.

    Returns:
        Feature frame with the same columns for development and scoring data.

    Raises:
        ValueError: If either date field contains an unparseable value.
    """
    origination = pd.to_datetime(frame["orig_date"], errors="coerce")
    maturity = pd.to_datetime(frame["end_date"], errors="coerce")
    require(origination.notna().all(), "orig_date contains unparseable values")
    require(maturity.notna().all(), "end_date contains unparseable values")

    features = pd.DataFrame(index=frame.index)

    # Monetary and size fields span orders of magnitude. Log transforms reduce
    # leverage from extreme values while preserving row order and missingness.
    features["log_loan_size"] = np.log1p(frame["loan_size_mm"])
    features["ltv_fraction"] = frame["ltv"] / 100
    features["occ_at_orig_fraction"] = frame["occ_at_orig"] / 100
    features["occ_fraction"] = frame["occ"] / 100
    features["occupancy_change"] = (frame["occ"] - frame["occ_at_orig"]) / 100

    # A signed log is used for NOI because current income can legitimately be
    # negative; an ordinary logarithm would discard or invalidate those records.
    features["signed_log_noi_orig"] = np.sign(frame["noi_at_orig"]) * np.log1p(
        frame["noi_at_orig"].abs()
    )
    features["signed_log_noi_current"] = np.sign(frame["noi"]) * np.log1p(
        frame["noi"].abs()
    )
    # The denominator floor prevents numerical instability, and fixed clipping
    # limits implausible ratio tails without learning thresholds from validation data.
    stable_noi_denominator = frame["noi_at_orig"].abs().clip(lower=1)
    features["noi_relative_change"] = (
        (frame["noi"] - frame["noi_at_orig"]) / stable_noi_denominator
    ).clip(-5, 5)
    features["is_30d"] = frame["is_30d"].astype(float)
    features["is_90d"] = frame["is_90d"].astype(float)
    features["loan_term_years"] = (maturity - origination).dt.days / 365.25
    features["origination_year"] = origination.dt.year.astype(float)

    # Negative ages indicate inconsistent source dates. Preserve that fact in a
    # flag, then mark the invalid numeric age missing for fold-fitted imputation.
    property_age = origination.dt.year - frame["year_built"]
    features["property_age_invalid"] = property_age.lt(0).astype(float)
    features["property_age"] = property_age.mask(property_age.lt(0))
    features["log_sqft"] = np.log1p(frame["sqft"])

    # Use stable, relatively low-cardinality categories. proptypelong is excluded
    # because it contains hundreds of inconsistent and scoring-only labels.
    features["cssaproptype"] = normalized_text(frame["cssaproptype"])
    features["msa_category"] = normalized_text(frame["msa_category"])
    features["state"] = normalized_text(frame["state"])
    features["office_group"] = classify_office_group(frame)
    return features


def build_preprocessor(
    numeric_features: list[str], *, dense_output: bool = False
) -> ColumnTransformer:
    """Build fold-fitted numeric and categorical preprocessing.

    Args:
        numeric_features: Engineered numeric columns to include in the model.
        dense_output: Whether one-hot encoding must return a dense matrix. Tree
            models require dense input; linear regression can retain sparse input.

    Returns:
        Unfitted column transformer that imputes, scales, and one-hot encodes data.
    """
    # Missing indicators let the model distinguish an imputed value from an
    # originally observed value. All statistics are learned inside the train fold.
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    # Unknown categories are ignored so legitimate scoring-only labels do not fail
    # inference. The broad categories used here have no missing values today, but
    # the imputer keeps the pipeline safe if future inputs do.
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense_output,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0 if dense_output else 0.3,
    )


def build_linear_pipeline(numeric_features: list[str]) -> Pipeline:
    """Build an unfitted ordinary least-squares pipeline.

    Args:
        numeric_features: Engineered numeric features to combine with the shared
            categorical feature set.

    Returns:
        Scikit-learn pipeline containing preprocessing and ``LinearRegression``.
    """
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_features)),
            ("model", LinearRegression()),
        ]
    )


def calculate_metrics(
    model_name: str,
    y_true: pd.Series | np.ndarray,
    predictions: np.ndarray,
    loan_weights: pd.Series | np.ndarray,
    *,
    target_policy: str = "raw supplied severity",
) -> dict[str, float | str]:
    """Calculate holdout metrics for one set of severity predictions.

    Args:
        model_name: Stable name written to the metrics table.
        y_true: Observed holdout severities.
        predictions: Model predictions aligned with ``y_true``.
        loan_weights: Loan balances used for the secondary exposure-weighted MSE.
        target_policy: Description of any target transformation used for this run.

    Returns:
        Dictionary containing required, diagnostic, and range-validity metrics.
    """
    y_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(predictions, dtype=float)
    weights = np.asarray(loan_weights, dtype=float)
    # The assignment requires unweighted R² and MSE. Balance-weighted MSE is kept
    # separate because it answers a portfolio-exposure question rather than the
    # assignment's equally weighted loan question.
    mse = mean_squared_error(y_array, pred_array)
    return {
        "model": model_name,
        "target_policy": target_policy,
        "r2": r2_score(y_array, pred_array),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_array, pred_array),
        "bias": float(np.mean(pred_array - y_array)),
        "loan_balance_weighted_mse": float(
            np.average(np.square(pred_array - y_array), weights=weights)
        ),
        "prediction_min": float(np.min(pred_array)),
        "prediction_max": float(np.max(pred_array)),
        "pct_predictions_below_zero": float(np.mean(pred_array < 0)),
        "pct_predictions_above_one": float(np.mean(pred_array > 1)),
    }


def subgroup_metrics(
    model_name: str,
    y_true: pd.Series,
    predictions: np.ndarray,
    validation_rows: pd.DataFrame,
) -> list[dict[str, float | int | str]]:
    """Measure model errors across important business subgroups.

    Args:
        model_name: Stable model label for the output table.
        y_true: Observed validation severities.
        predictions: Model predictions aligned with ``y_true``.
        validation_rows: Raw validation rows used to derive subgroup labels.

    Returns:
        Long-form metric records for loss status, delinquency, and office exposure.
    """
    rows: list[dict[str, float | int | str]] = []
    pred_series = pd.Series(predictions, index=y_true.index)
    # Loss/no-loss reveals behavior at the target's point mass; delinquency and
    # office exposure reveal performance in the most decision-relevant segments.
    dimensions = {
        "actual_loss_status": np.where(
            y_true.gt(0), "positive severity", "zero severity"
        ),
        "delinquency_status": np.select(
            [validation_rows["is_90d"], validation_rows["is_30d"]],
            ["90+ days", "30–89 days"],
            default="not 30 days delinquent",
        ),
        "office_group": classify_office_group(validation_rows),
    }
    for dimension, labels in dimensions.items():
        label_series = pd.Series(labels, index=y_true.index)
        for subgroup in sorted(label_series.unique()):
            mask = label_series.eq(subgroup)
            actual = y_true.loc[mask]
            predicted = pred_series.loc[mask]
            rows.append(
                {
                    "model": model_name,
                    "dimension": dimension,
                    "subgroup": subgroup,
                    "n": int(mask.sum()),
                    "actual_mean": float(actual.mean()),
                    "predicted_mean": float(predicted.mean()),
                    "mse": mean_squared_error(actual, predicted),
                    "mae": mean_absolute_error(actual, predicted),
                }
            )
    return rows


def plot_modeling_eda(development: pd.DataFrame, output_path: Path) -> None:
    """Create a compact visual audit of modeling-readiness concerns.

    Args:
        development: Merged labeled development rows.
        output_path: Destination PNG path.
    """
    target = development[TARGET_COLUMN]
    positive = target[target.gt(0)]
    missing = development.isna().sum()
    missing = missing[missing.gt(0)].sort_values(ascending=False)
    delinquency = np.select(
        [development["is_90d"], development["is_30d"]],
        ["90+ days", "30–89 days"],
        default="Not 30 days delinquent",
    )
    delinquency_order = ["Not 30 days delinquent", "30–89 days", "90+ days"]
    delinquency_summary = (
        development.assign(delinquency_status=delinquency)
        .groupby("delinquency_status")[TARGET_COLUMN]
        .agg(
            mean="mean", positive_share=lambda values: values.gt(0).mean(), count="size"
        )
        .reindex(delinquency_order)
    )

    # The four panels answer separate readiness questions: target imbalance, target
    # tail behavior, missing-data concentration, and possible timing/leakage signal.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Part 2 modeling-readiness EDA",
        fontsize=18,
        fontweight="bold",
        x=0.06,
        ha="left",
    )

    target_counts = pd.Series(
        [target.eq(0).sum(), target.gt(0).sum()],
        index=["Zero severity", "Positive severity"],
    )
    bars = axes[0, 0].bar(
        target_counts.index, target_counts.values, color=["#64748B", "#2563EB"]
    )
    axes[0, 0].set_title(
        "Severity has a large point mass at zero", loc="left", fontweight="bold"
    )
    axes[0, 0].set_ylabel("Loans")
    for bar, value in zip(bars, target_counts.values):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n({value / len(target):.1%})",
            ha="center",
            va="bottom",
            fontweight="semibold",
        )

    # Clip only the plotted copy so the 13.65 outlier does not flatten the visible
    # distribution. Modeling always uses the unmodified supplied target.
    axes[0, 1].hist(positive.clip(upper=1.5), bins=45, color="#7C3AED", alpha=0.9)
    axes[0, 1].axvline(1, color="#DC2626", linestyle="--", linewidth=1.4)
    axes[0, 1].set_title(
        "Positive severity is right-skewed", loc="left", fontweight="bold"
    )
    axes[0, 1].set_xlabel("Severity (values above 1.5 shown at boundary)")
    axes[0, 1].set_ylabel("Loans")
    axes[0, 1].text(
        0.98,
        0.95,
        f"{target.gt(1).sum()} values > 1\nmaximum = {target.max():.2f}",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
        },
    )

    axes[1, 0].barh(missing.index[::-1], missing.values[::-1], color="#0F766E")
    axes[1, 0].set_title(
        "Missingness is concentrated in property size and occupancy",
        loc="left",
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("Missing rows")
    for index, value in enumerate(missing.values[::-1]):
        axes[1, 0].text(value, index, f" {value:,}", va="center", fontsize=9)

    bars = axes[1, 1].bar(
        delinquency_summary.index,
        delinquency_summary["mean"],
        color=["#94A3B8", "#F59E0B", "#DC2626"],
    )
    axes[1, 1].set_title(
        "Current delinquency is highly informative", loc="left", fontweight="bold"
    )
    axes[1, 1].set_ylabel("Mean severity")
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].tick_params(axis="x", rotation=12)
    for bar, (_, row) in zip(bars, delinquency_summary.iterrows()):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            row["mean"] + 0.008,
            f"{row['mean']:.1%}\npositive: {row['positive_share']:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
        )

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#CBD5E1", alpha=0.6)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_linear_scatter(
    y_true: pd.Series,
    predictions: np.ndarray,
    metrics: dict[str, float | str],
    output_path: Path,
) -> None:
    """Create the required true-versus-linear-predicted scatter plot.

    Args:
        y_true: Observed validation severities.
        predictions: Linear-regression predictions aligned with ``y_true``.
        metrics: Metric record for the current-state linear model.
        output_path: Destination PNG path.
    """
    y_array = y_true.to_numpy(dtype=float)
    lower = min(-0.05, float(predictions.min()))
    upper = max(1.05, float(y_array.max()), float(predictions.max()))

    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    ax.scatter(
        y_array,
        predictions,
        s=14,
        alpha=0.22,
        color="#2563EB",
        edgecolors="none",
        rasterized=True,
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="#DC2626",
        linewidth=1.5,
        label="Perfect prediction",
    )
    ax.axhline(0, color="#64748B", linewidth=0.9)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("True severity")
    ax.set_ylabel("Linear-regression predicted severity")
    ax.set_title(
        "Linear regression misses the zero mass and produces negative severities",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="upper left", frameon=False)
    ax.grid(color="#CBD5E1", alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    metric_text = (
        f"Grouped holdout (n={len(y_true):,})\n"
        f"R² = {metrics['r2']:.3f}\n"
        f"MSE = {metrics['mse']:.4f}\n"
        f"MAE = {metrics['mae']:.4f}\n"
        f"Predictions < 0: {metrics['pct_predictions_below_zero']:.1%}"
    )
    ax.text(
        0.98,
        0.04,
        metric_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
        },
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


class HurdleSeverityModel:
    """Two-stage model for zero-inflated severity.

    Stage 1 predicts whether severity is positive. Stage 2 predicts severity
    conditional on a positive loss. Their product is expected severity.

    Attributes:
        preprocessor: Fold-fitted numeric/categorical transformer.
        occurrence_model: Classifier for the event ``severity > 0``.
        magnitude_model: Positive-target regressor for conditional severity.
    """

    def __init__(self, seed: int) -> None:
        """Initialize deterministic preprocessing and both hurdle stages.

        Args:
            seed: Random seed shared by both gradient-boosted models.
        """
        self.preprocessor = build_preprocessor(
            CURRENT_STATE_NUMERIC_FEATURES, dense_output=True
        )
        self.occurrence_model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=220,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )
        self.magnitude_model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=220,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "HurdleSeverityModel":
        """Fit preprocessing, loss occurrence, and positive-loss magnitude.

        Args:
            features: Engineered development features.
            target: Nonnegative severity target aligned with ``features``.

        Returns:
            The fitted model instance.

        Raises:
            ValueError: If the training target lacks either zero or positive rows.
        """
        matrix = self.preprocessor.fit_transform(features)
        has_loss = target.gt(0)
        require(has_loss.any(), "Hurdle training requires positive-severity rows")
        require((~has_loss).any(), "Hurdle training requires zero-severity rows")

        # The occurrence stage is deliberately fit without class reweighting so its
        # probability can be multiplied directly by conditional expected severity.
        self.occurrence_model.fit(matrix, has_loss.astype(int))

        # Poisson loss applies a log link and guarantees positive conditional means.
        self.magnitude_model.fit(matrix[has_loss.to_numpy()], target.loc[has_loss])
        return self

    def predict_components(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict loss probability and conditional positive-loss magnitude.

        Args:
            features: Engineered rows to score.

        Returns:
            Tuple of positive-loss probabilities and conditional severity estimates.
        """
        matrix = self.preprocessor.transform(features)
        probability = self.occurrence_model.predict_proba(matrix)[:, 1]
        magnitude = np.maximum(self.magnitude_model.predict(matrix), 0)
        return probability, magnitude

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict nonnegative expected severity.

        Args:
            features: Engineered rows to score.

        Returns:
            Expected severity, computed as probability times conditional magnitude.
        """
        probability, magnitude = self.predict_components(features)
        return probability * magnitude


@dataclass(frozen=True)
class Holdout:
    """Property-grouped development and validation partitions.

    Attributes:
        x_train: Engineered features used for fitting.
        x_validation: Engineered features reserved for evaluation.
        y_train: Training severities.
        y_validation: Validation severities.
        validation_rows: Raw validation rows for weights and subgroup analysis.
    """

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    validation_rows: pd.DataFrame

    @property
    def validation_weights(self) -> pd.Series:
        """Return loan balances aligned with the validation observations."""
        return self.validation_rows["loan_size_mm"]


@dataclass
class ModelResults:
    """Models, predictions, and tables produced during holdout evaluation.

    Attributes:
        linear_model: Fitted required current-state linear pipeline.
        hurdle_model: Fitted two-stage current-state model.
        linear_predictions: Linear holdout predictions.
        hurdle_predictions: Hurdle holdout predictions.
        metrics: Overall comparison table for all baselines and models.
        subgroup_metrics: Long-form diagnostic metrics by business segment.
        hurdle_metrics: Occurrence and conditional-magnitude component metrics.
    """

    linear_model: Pipeline
    hurdle_model: HurdleSeverityModel
    linear_predictions: np.ndarray
    hurdle_predictions: np.ndarray
    metrics: pd.DataFrame
    subgroup_metrics: pd.DataFrame
    hurdle_metrics: dict[str, float]


def print_metric_table(metrics_frame: pd.DataFrame) -> None:
    """Print the model comparison using stable columns and precision.

    Args:
        metrics_frame: Overall metric records returned by ``evaluate_models``.
    """
    display_columns = [
        "model",
        "target_policy",
        "r2",
        "mse",
        "rmse",
        "mae",
        "prediction_min",
        "prediction_max",
        "pct_predictions_below_zero",
        "pct_predictions_above_one",
    ]
    printable = metrics_frame[display_columns].copy()
    numeric = printable.select_dtypes(include="number").columns
    printable[numeric] = printable[numeric].round(6)
    print(printable.to_string(index=False))


def report_eda(
    loans: pd.DataFrame,
    development: pd.DataFrame,
    scoring: pd.DataFrame,
    metric_dir: Path,
) -> None:
    """Print and save the checks that determine the modeling design.

    Args:
        loans: Raw labeled loan rows, used to audit repeated properties.
        development: Merged labeled development rows.
        scoring: Merged unlabeled scoring rows.
        metric_dir: Directory where the feature-audit CSV is written.
    """
    print_section("1. DATA CONTRACT AND MODELING-READINESS EDA")
    print(
        f"Development: {len(development):,} loans, "
        f"{development['propname'].nunique():,} properties; "
        f"scoring: {len(scoring):,} loans. All many-to-one property joins matched."
    )
    print(
        f"Target: {development[TARGET_COLUMN].eq(0).mean():.2%} zero, "
        f"{development[TARGET_COLUMN].gt(0).mean():.2%} positive, "
        f"{development[TARGET_COLUMN].gt(1).sum():,} above 1, "
        f"maximum {development[TARGET_COLUMN].max():.4f}."
    )
    repeated_properties = loans.groupby("propname").size().gt(1).sum()
    train_score_property_overlap = len(
        set(development["propname"]) & set(scoring["propname"])
    )
    print(
        f"Grouping risk: {repeated_properties:,} development properties back "
        "multiple loans; "
        f"{train_score_property_overlap:,} scoring properties also occur in "
        "development."
    )

    feature_audit = make_feature_audit(development, scoring)
    feature_audit.to_csv(metric_dir / "part2_feature_audit.csv", index=False)
    print("\nColumns with missing values or unseen scoring categories:")
    print(
        feature_audit.loc[
            feature_audit["train_missing"].gt(0)
            | feature_audit["score_missing"].gt(0)
            | feature_audit["unseen_score_categories"].gt(0)
        ].to_string(index=False)
    )

    origination = pd.to_datetime(development["orig_date"])
    property_age = origination.dt.year - development["year_built"]
    print(
        f"Temporal checks: origination {origination.min().date()} to "
        f"{origination.max().date()}; {property_age.lt(0).sum():,} negative "
        "property ages are flagged and imputed."
    )
    print(
        "Leakage audit: loan_id and propname are excluded. Current occ/noi and "
        "delinquency are included only under the stated current-state assumption; "
        "an origination-only linear model measures sensitivity to that assumption."
    )


def make_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    development: pd.DataFrame,
    seed: int,
) -> Holdout:
    """Create a deterministic validation split with no property overlap.

    Args:
        features: Engineered features for all labeled rows.
        target: Severity target aligned with ``features``.
        development: Raw merged rows containing the property grouping key.
        seed: Random seed used by ``GroupShuffleSplit``.

    Returns:
        Training and validation partitions with raw validation metadata.

    Raises:
        ValueError: If properties cross partitions or loss prevalence differs by
        more than two percentage points.
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VALIDATION_SIZE, random_state=seed
    )
    train_index, validation_index = next(
        splitter.split(features, target, groups=development["propname"])
    )
    validation_rows = development.iloc[validation_index]
    train_properties = set(development.iloc[train_index]["propname"])
    validation_properties = set(validation_rows["propname"])
    require(
        train_properties.isdisjoint(validation_properties),
        "A property appears in both training and validation partitions",
    )
    y_train = target.iloc[train_index]
    y_validation = target.iloc[validation_index]
    require(
        abs(y_train.gt(0).mean() - y_validation.gt(0).mean()) < 0.02,
        "Train/validation positive-severity rates differ by at least 2 "
        "percentage points",
    )

    return Holdout(
        x_train=features.iloc[train_index],
        x_validation=features.iloc[validation_index],
        y_train=y_train,
        y_validation=y_validation,
        validation_rows=validation_rows,
    )


def report_holdout(holdout: Holdout) -> None:
    """Print the rationale and key checks for the validation design.

    Args:
        holdout: Property-grouped training and validation partitions.
    """
    train_count = len(holdout.y_train)
    validation_count = len(holdout.y_validation)

    print_section("2. VALIDATION DESIGN")
    print(
        f"Property-grouped holdout: {train_count:,} training loans and "
        f"{validation_count:,} validation loans."
    )
    print(
        f"Positive-severity share: train={holdout.y_train.gt(0).mean():.2%}, "
        f"validation={holdout.y_validation.gt(0).mean():.2%}. Property overlap=0."
    )
    print(
        "A random row split is not used because it could place loans backed by the "
        "same property in both partitions. A chronological sensitivity split would "
        "be useful for future-cohort deployment, but no severity observation date "
        "is supplied."
    )


def constant_baseline_predictions(holdout: Holdout) -> dict[str, np.ndarray]:
    """Return the two simplest honest prediction benchmarks.

    Args:
        holdout: Property-grouped training and validation partitions.

    Returns:
        Predictions from an all-zero rule and a training-mean regressor.
    """
    zero = np.zeros(len(holdout.y_validation))
    mean_model = DummyRegressor(strategy="mean")
    mean_model.fit(np.zeros((len(holdout.y_train), 1)), holdout.y_train)
    mean = mean_model.predict(np.zeros((len(holdout.y_validation), 1)))
    return {"all_zero_baseline": zero, "training_mean_baseline": mean}


def evaluate_models(holdout: Holdout, seed: int) -> ModelResults:
    """Fit and evaluate every baseline and candidate on one holdout.

    Args:
        holdout: Property-grouped training and validation partitions.
        seed: Random seed used by the hurdle model.

    Returns:
        Fitted primary models, holdout predictions, and diagnostic metric tables.
    """
    metric_rows: list[dict[str, float | str]] = []
    subgroup_rows: list[dict[str, float | int | str]] = []

    # Constant baselines establish whether a fitted model adds value beyond target
    # prevalence and the unconditional mean.
    for name, predictions in constant_baseline_predictions(holdout).items():
        metric_rows.append(
            calculate_metrics(
                name,
                holdout.y_validation,
                predictions,
                holdout.validation_weights,
            )
        )

    # This sensitivity model quantifies how much apparent signal comes from updated
    # fields that would be unavailable in an at-origination forecasting use case.
    origination_linear = build_linear_pipeline(ORIGINATION_NUMERIC_FEATURES)
    origination_linear.fit(holdout.x_train, holdout.y_train)
    origination_predictions = origination_linear.predict(holdout.x_validation)
    metric_rows.append(
        calculate_metrics(
            "linear_origination_only",
            holdout.y_validation,
            origination_predictions,
            holdout.validation_weights,
        )
    )

    # The required OLS model uses current-state fields under the stated as-of-date
    # assumption and remains the primary assignment baseline.
    current_linear = build_linear_pipeline(CURRENT_STATE_NUMERIC_FEATURES)
    current_linear.fit(holdout.x_train, holdout.y_train)
    linear_predictions = current_linear.predict(holdout.x_validation)
    linear_metrics = calculate_metrics(
        "linear_current_state",
        holdout.y_validation,
        linear_predictions,
        holdout.validation_weights,
    )
    metric_rows.append(linear_metrics)
    subgroup_rows.extend(
        subgroup_metrics(
            "linear_current_state",
            holdout.y_validation,
            linear_predictions,
            holdout.validation_rows,
        )
    )

    # Do not silently alter the supplied target. This separately labeled run shows
    # whether the 108 values above one materially change the OLS conclusion.
    capped_linear = build_linear_pipeline(CURRENT_STATE_NUMERIC_FEATURES)
    capped_linear.fit(holdout.x_train, holdout.y_train.clip(0, 1))
    capped_predictions = capped_linear.predict(holdout.x_validation)
    metric_rows.append(
        calculate_metrics(
            "linear_current_state_capped_sensitivity",
            holdout.y_validation.clip(0, 1),
            capped_predictions,
            holdout.validation_weights,
            target_policy="severity clipped to [0, 1]",
        )
    )

    # The hurdle experiment matches the target-generating structure: occurrence of
    # any loss followed by continuous loss magnitude conditional on occurrence.
    hurdle_model = HurdleSeverityModel(seed).fit(holdout.x_train, holdout.y_train)
    positive_probability, conditional_magnitude = hurdle_model.predict_components(
        holdout.x_validation
    )
    hurdle_predictions = positive_probability * conditional_magnitude
    hurdle_model_metrics = calculate_metrics(
        "hurdle_current_state",
        holdout.y_validation,
        hurdle_predictions,
        holdout.validation_weights,
    )
    metric_rows.append(hurdle_model_metrics)
    subgroup_rows.extend(
        subgroup_metrics(
            "hurdle_current_state",
            holdout.y_validation,
            hurdle_predictions,
            holdout.validation_rows,
        )
    )

    has_loss = holdout.y_validation.gt(0)
    hurdle_component_metrics = {
        "roc_auc": roc_auc_score(has_loss, positive_probability),
        "average_precision": average_precision_score(has_loss, positive_probability),
        "brier_score": brier_score_loss(has_loss, positive_probability),
        "positive_magnitude_mae": mean_absolute_error(
            holdout.y_validation.loc[has_loss],
            conditional_magnitude[has_loss.to_numpy()],
        ),
    }

    return ModelResults(
        linear_model=current_linear,
        hurdle_model=hurdle_model,
        linear_predictions=linear_predictions,
        hurdle_predictions=hurdle_predictions,
        metrics=pd.DataFrame(metric_rows),
        subgroup_metrics=pd.DataFrame(subgroup_rows),
        hurdle_metrics=hurdle_component_metrics,
    )


def report_linear_results(results: ModelResults) -> None:
    """Print the assignment-required linear-regression outputs.

    Args:
        results: Completed holdout evaluation results.
    """
    linear_metrics = results.metrics.set_index("model").loc["linear_current_state"]
    print_section("3. REQUIRED LINEAR REGRESSION")
    print("Included current-state feature rationale:")
    for feature in CURRENT_STATE_NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        print(f"- {feature}: {FEATURE_RATIONALES[feature]}")
    print(
        f"\nLINEAR REGRESSION R²: {linear_metrics['r2']:.6f}\n"
        f"LINEAR REGRESSION MSE: {linear_metrics['mse']:.6f}\n"
        f"LINEAR REGRESSION RMSE: {linear_metrics['rmse']:.6f}\n"
        f"LINEAR REGRESSION MAE: {linear_metrics['mae']:.6f}"
    )
    print(
        f"Prediction range: {linear_metrics['prediction_min']:.4f} to "
        f"{linear_metrics['prediction_max']:.4f}; "
        f"{linear_metrics['pct_predictions_below_zero']:.2%} are negative."
    )


def report_model_comparison(results: ModelResults) -> str:
    """Print model comparison and choose the model used for scoring.

    Args:
        results: Completed holdout evaluation results.

    Returns:
        Stable name of the raw-target candidate with the lowest holdout MSE.
    """
    print_section("4. MODEL COMPARISON")
    print_metric_table(results.metrics)
    component = results.hurdle_metrics
    print(
        "\nHurdle occurrence metrics: "
        f"ROC-AUC={component['roc_auc']:.4f}, "
        f"average precision={component['average_precision']:.4f}, "
        f"Brier score={component['brier_score']:.4f}."
    )
    print(
        f"Positive-case conditional magnitude MAE: "
        f"{component['positive_magnitude_mae']:.4f}."
    )

    # Selection is intentionally restricted to models fit on the raw target. The
    # capped sensitivity run answers a data-policy question and is not comparable.
    candidate_names = ["linear_current_state", "hurdle_current_state"]
    candidate_metrics = results.metrics.set_index("model").loc[candidate_names, "mse"]
    selected_model = str(candidate_metrics.idxmin())
    print(f"Selected for scoring by lowest raw-target validation MSE: {selected_model}")
    return selected_model


def save_evaluation_outputs(results: ModelResults, metric_dir: Path) -> None:
    """Persist metrics used by the written response and QA checks.

    Args:
        results: Completed holdout evaluation results.
        metric_dir: Directory where metric CSVs are written.
    """
    results.metrics.to_csv(metric_dir / "part2_metrics.csv", index=False)
    results.subgroup_metrics.to_csv(
        metric_dir / "part2_subgroup_metrics.csv", index=False
    )
    pd.DataFrame([results.hurdle_metrics]).to_csv(
        metric_dir / "part2_hurdle_component_metrics.csv", index=False
    )


def score_predictions(
    features: pd.DataFrame,
    target: pd.Series,
    scoring_features: pd.DataFrame,
    scoring_rows: pd.DataFrame,
    prediction_path: Path,
    output_dir: Path,
    metric_dir: Path,
    selected_model: str,
    seed: int,
) -> None:
    """Refit raw-target candidates and score the supplied prediction rows.

    Args:
        features: Engineered features for all labeled development rows.
        target: Full labeled severity target.
        scoring_features: Engineered features for the unlabeled scoring rows.
        scoring_rows: Raw merged scoring rows used to preserve identifiers.
        prediction_path: Original scoring CSV, which remains read-only.
        output_dir: Directory where the scored CSV is written.
        metric_dir: Directory where model-comparison predictions are written.
        selected_model: Candidate name selected on holdout MSE.
        seed: Random seed used by the hurdle model.

    Raises:
        ValueError: If the output count, finiteness, model name, or row identity is
        inconsistent with the source scoring data.
    """
    print_section("5. REFIT AND SCORE predictions.csv")

    linear_model = build_linear_pipeline(CURRENT_STATE_NUMERIC_FEATURES)
    linear_model.fit(features, target)
    linear_predictions = linear_model.predict(scoring_features)

    hurdle_model = HurdleSeverityModel(seed).fit(features, target)
    hurdle_predictions = hurdle_model.predict(scoring_features)

    candidate_predictions = {
        "linear_current_state": linear_predictions,
        "hurdle_current_state": hurdle_predictions,
    }
    require(
        selected_model in candidate_predictions,
        f"Unknown scoring model: {selected_model}",
    )
    selected_predictions = candidate_predictions[selected_model]

    # Reload the untouched source file so the deliverable retains exact source order
    # and non-target columns rather than depending on a merged modeling frame.
    scored_output = pd.read_csv(prediction_path)
    require(
        len(selected_predictions) == len(scoring_rows) == len(scored_output),
        "Scoring prediction count does not match the source row count",
    )
    require(
        np.isfinite(selected_predictions).all(),
        "Scoring predictions contain missing or infinite values",
    )
    scored_output[TARGET_COLUMN] = selected_predictions
    require(
        scored_output["loan_id"].equals(scoring_rows["loan_id"].reset_index(drop=True)),
        "Scored loan IDs do not preserve source order",
    )
    scored_output.to_csv(output_dir / "predictions_scored.csv", index=False)

    comparison_output = scoring_rows[["loan_id", "propname"]].copy()
    comparison_output["linear_severity"] = linear_predictions
    comparison_output["hurdle_severity"] = hurdle_predictions
    comparison_output["selected_model"] = selected_model
    comparison_output.to_csv(
        metric_dir / "part2_scoring_model_comparison.csv", index=False
    )
    print(
        f"Wrote {len(scored_output):,} scored rows to "
        f"{output_dir / 'predictions_scored.csv'}."
    )
    print(
        f"Selected prediction range: {selected_predictions.min():.4f} to "
        f"{selected_predictions.max():.4f}; "
        f"missing={np.isnan(selected_predictions).sum()}."
    )
    print("Input CSV files were read only and were not overwritten.")


def main() -> None:
    """Run EDA, holdout evaluation, artifact generation, and final scoring."""
    args = parse_args()
    figure_dir = args.output_dir / "figures"
    metric_dir = args.output_dir / "metrics"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    print_section("PART 2 — MODELING LOSS")
    print(
        f"pandas={pd.__version__}; numpy={np.__version__}; "
        f"scikit-learn={sklearn.__version__}"
    )
    print(f"Random seed: {args.seed}")
    print(
        "Prediction timing assumption: current-state estimation. Updated occupancy, "
        "NOI, and delinquency are treated as available because they are populated "
        "in predictions.csv."
    )

    # Load once, then route the same validated frames through EDA and modeling.
    loans, _properties, development, scoring = load_and_validate(
        args.loan_data, args.property_data, args.prediction_data
    )
    report_eda(loans, development, scoring, metric_dir)
    plot_modeling_eda(development, figure_dir / "part2_modeling_eda.png")

    # The identical feature builder is applied to labeled and scoring rows before
    # any estimator-specific preprocessing is fitted.
    features = engineer_features(development)
    scoring_features = engineer_features(scoring)
    require(
        features.columns.tolist() == scoring_features.columns.tolist(),
        "Development and scoring feature schemas differ",
    )
    numeric = features.drop(columns=CATEGORICAL_FEATURES)
    require(
        not numeric.replace([np.inf, -np.inf], np.nan).isna().all(axis=0).any(),
        "At least one engineered numeric feature is entirely non-finite",
    )
    target = development[TARGET_COLUMN]

    holdout = make_holdout(features, target, development, args.seed)
    report_holdout(holdout)
    results = evaluate_models(holdout, args.seed)
    report_linear_results(results)
    linear_metrics = results.metrics.set_index("model").loc["linear_current_state"]
    plot_linear_scatter(
        holdout.y_validation,
        results.linear_predictions,
        linear_metrics,
        figure_dir / "part2_linear_actual_vs_predicted.png",
    )

    save_evaluation_outputs(results, metric_dir)
    selected_model = report_model_comparison(results)
    score_predictions(
        features,
        target,
        scoring_features,
        scoring,
        args.prediction_data,
        args.output_dir,
        metric_dir,
        selected_model,
        args.seed,
    )


if __name__ == "__main__":
    main()
