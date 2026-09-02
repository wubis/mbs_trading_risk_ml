#!/usr/bin/env python3
"""Fit the required severity regression, compare a hurdle model, and score loans.

The workflow has six stages:
1. Load the loan and property files and validate their data contract.
2. Build the same model features for development and scoring rows.
3. Create a property-grouped holdout so one property cannot cross the split.
4. Evaluate simple baselines, the required linear model, and a hurdle model.
5. Check model stability with repeated property-grouped cross-validation.
6. Refit the selected model on all labeled rows and score predictions.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Configuration, expected source schemas, and model feature sets
# -----------------------------------------------------------------------------

RANDOM_SEED = 20260901
VALIDATION_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 3

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
CATEGORICAL_FEATURES = ["cssaproptype", "msa_category", "state", "office_group"]


# -----------------------------------------------------------------------------
# Command-line and validation helpers
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Define paths and random seed while keeping repository-root defaults."""
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


def require(condition: bool, message: str) -> None:
    """Raise a readable data-contract error when a required condition fails."""
    if not condition:
        raise ValueError(message)


def normalized_text(series: pd.Series) -> pd.Series:
    """Normalize category labels without replacing missing values."""
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .str.replace(r"[\s_-]+", " ", regex=True)
    )


# -----------------------------------------------------------------------------
# Data loading and feature engineering
# -----------------------------------------------------------------------------

def load_and_validate(
    loan_path: Path, property_path: Path, prediction_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load inputs and enforce the schema, key, target, and merge contracts."""
    loans = pd.read_csv(loan_path)
    properties = pd.read_csv(property_path)
    scoring = pd.read_csv(prediction_path)

    # Fail early if the supplied files no longer match the assignment contract.
    require(loans.columns.tolist() == EXPECTED_LOAN_COLUMNS, "Unexpected loan schema")
    require(
        scoring.columns.tolist() == EXPECTED_LOAN_COLUMNS, "Unexpected scoring schema"
    )
    require(
        properties.columns.tolist() == EXPECTED_PROPERTY_COLUMNS,
        "Unexpected property schema",
    )
    require(loans["loan_id"].is_unique, "Training loan_id must be unique")
    require(scoring["loan_id"].is_unique, "Scoring loan_id must be unique")
    require(
        set(loans["loan_id"]).isdisjoint(scoring["loan_id"]),
        "Train/scoring loan IDs overlap",
    )
    require(properties["propname"].is_unique, "Property propname must be unique")
    require(loans["severity"].notna().all(), "Training severity must be populated")
    require(scoring["severity"].isna().all(), "Scoring severity must be blank")
    require(loans["severity"].ge(0).all(), "Severity must be nonnegative")
    require((~loans["is_90d"] | loans["is_30d"]).all(), "is_90d must imply is_30d")

    # One property can back several loans, but each loan must match exactly one
    # property row. The indicator makes unmatched keys explicit.
    def merge(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        merged = frame.merge(
            properties,
            on="propname",
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        require(len(merged) == len(frame), f"{label} merge changed row count")
        require(merged["_merge"].eq("both").all(), f"Unmatched {label} properties")
        return merged.drop(columns="_merge")

    return merge(loans, "development"), merge(scoring, "scoring"), scoring


def classify_office_group(frame: pd.DataFrame) -> np.ndarray:
    """Assign the Part 1 office-exposure grouping for use as a model feature."""
    cssa = normalized_text(frame["cssaproptype"])
    broad_type = normalized_text(frame["proptype"])
    detailed_type = normalized_text(frame["proptypelong"])
    is_office = cssa.eq("of")
    is_mixed = cssa.eq("mu") | broad_type.eq("mixed use")
    has_office = detailed_type.str.contains(r"\boffice\b", regex=True, na=False)
    unknown_mixed = is_mixed & detailed_type.isna()
    return np.select(
        [is_office, ~is_office & has_office, unknown_mixed],
        ["office", "mixed office", "unknown mixed"],
        default="non office",
    )


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the same economically motivated features for training and scoring."""
    # Dates become model-friendly term, vintage, and property-age features.
    origination = pd.to_datetime(frame["orig_date"], errors="coerce")
    maturity = pd.to_datetime(frame["end_date"], errors="coerce")
    require(origination.notna().all(), "orig_date contains unparseable values")
    require(maturity.notna().all(), "end_date contains unparseable values")

    features = pd.DataFrame(index=frame.index)

    # Loan structure and current operating condition. Log transforms reduce the
    # influence of very large loans and NOI values while signed logs retain losses.
    features["log_loan_size"] = np.log1p(frame["loan_size_mm"])
    features["ltv_fraction"] = frame["ltv"] / 100
    features["occ_at_orig_fraction"] = frame["occ_at_orig"] / 100
    features["occ_fraction"] = frame["occ"] / 100
    features["occupancy_change"] = (frame["occ"] - frame["occ_at_orig"]) / 100
    features["signed_log_noi_orig"] = np.sign(frame["noi_at_orig"]) * np.log1p(
        frame["noi_at_orig"].abs()
    )
    features["signed_log_noi_current"] = np.sign(frame["noi"]) * np.log1p(
        frame["noi"].abs()
    )
    # A denominator floor avoids division by values near zero. The final clipping
    # prevents a few extreme changes from dominating the linear regression.
    noi_denominator = frame["noi_at_orig"].abs().clip(lower=1)
    features["noi_relative_change"] = (
        (frame["noi"] - frame["noi_at_orig"]) / noi_denominator
    ).clip(-5, 5)
    features["is_30d"] = frame["is_30d"].astype(float)
    features["is_90d"] = frame["is_90d"].astype(float)
    features["loan_term_years"] = (maturity - origination).dt.days / 365.25
    features["origination_year"] = origination.dt.year.astype(float)

    # Negative implied ages are data-quality problems: retain that information in
    # a flag, then let the preprocessing pipeline impute the invalid age itself.
    property_age = origination.dt.year - frame["year_built"]
    features["property_age_invalid"] = property_age.lt(0).astype(float)
    features["property_age"] = property_age.mask(property_age.lt(0))
    features["log_sqft"] = np.log1p(frame["sqft"])

    # Normalize categories once so training and scoring use identical labels.
    features["cssaproptype"] = normalized_text(frame["cssaproptype"])
    features["msa_category"] = normalized_text(frame["msa_category"])
    features["state"] = normalized_text(frame["state"])
    features["office_group"] = classify_office_group(frame)
    return features


# -----------------------------------------------------------------------------
# Preprocessing and model definitions
# -----------------------------------------------------------------------------

def build_preprocessor(
    numeric_features: list[str], *, dense_output: bool = False
) -> ColumnTransformer:
    """Build fold-safe numeric and categorical preprocessing."""
    # Missingness indicators preserve whether an originally observed value was
    # absent; medians and scaling are learned only from the training partition.
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    # Unknown-category handling allows scoring rows to contain labels not seen in
    # model development. Tree models request dense output below.
    categorical = Pipeline(
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
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0 if dense_output else 0.3,
    )


def build_linear_pipeline(numeric_features: list[str]) -> Pipeline:
    """Combine preprocessing with the assignment's required OLS model."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_features)),
            ("model", LinearRegression()),
        ]
    )


class HurdleSeverityModel:
    """Model bounded expected severity as probability times positive-loss magnitude."""

    def __init__(self, seed: int) -> None:
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

    def fit(self, features: pd.DataFrame, target: pd.Series) -> HurdleSeverityModel:
        """Fit loss occurrence on all rows and magnitude on positive rows only."""
        matrix = self.preprocessor.fit_transform(features)
        has_loss = target.gt(0)
        require(has_loss.any(), "Hurdle training requires positive losses")
        require((~has_loss).any(), "Hurdle training requires zero losses")
        self.occurrence_model.fit(matrix, has_loss.astype(int))
        self.magnitude_model.fit(matrix[has_loss.to_numpy()], target.loc[has_loss])
        return self

    def predict_components(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return probability of loss and severity conditional on a loss."""
        matrix = self.preprocessor.transform(features)
        probability = self.occurrence_model.predict_proba(matrix)[:, 1]
        magnitude = np.clip(self.magnitude_model.predict(matrix), 0, 1)
        return probability, magnitude

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return expected severity: loss probability times conditional severity."""
        probability, magnitude = self.predict_components(features)
        return probability * magnitude


# -----------------------------------------------------------------------------
# Holdout evaluation and metrics
# -----------------------------------------------------------------------------

def calculate_metrics(
    model_name: str,
    y_true: pd.Series | np.ndarray,
    predictions: np.ndarray,
    loan_weights: pd.Series | np.ndarray,
    *,
    target_policy: str = "severity clipped to [0, 1]",
) -> dict[str, float | str]:
    """Calculate fit, error, bias, weighting, and prediction-range diagnostics."""
    y_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(predictions, dtype=float)
    weights = np.asarray(loan_weights, dtype=float)
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


def make_holdout(
    features: pd.DataFrame, target: pd.Series, groups: pd.Series, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create one 80/20 split with no property appearing on both sides."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VALIDATION_SIZE, random_state=seed
    )
    train_index, validation_index = next(
        splitter.split(features, target, groups=groups)
    )
    require(
        set(groups.iloc[train_index]).isdisjoint(groups.iloc[validation_index]),
        "A property crosses the holdout boundary",
    )
    positive_gap = abs(
        target.iloc[train_index].gt(0).mean()
        - target.iloc[validation_index].gt(0).mean()
    )
    require(positive_gap < 0.02, "Holdout positive-severity rates differ materially")
    return train_index, validation_index


def evaluate_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    raw_target: pd.Series,
    development: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    seed: int,
) -> dict[str, object]:
    """Fit all headline models on one shared property-grouped holdout."""
    # Every model sees the same train/validation rows, enabling paired comparison.
    x_train = features.iloc[train_index]
    x_validation = features.iloc[validation_index]
    y_train = target.iloc[train_index]
    y_validation = target.iloc[validation_index]
    raw_y_train = raw_target.iloc[train_index]
    raw_y_validation = raw_target.iloc[validation_index]
    weights = development.iloc[validation_index]["loan_size_mm"]
    metric_rows: list[dict[str, float | str]] = []

    # Simple references show whether a fitted model beats naive severity guesses.
    zero_predictions = np.zeros(len(y_validation))
    metric_rows.append(
        calculate_metrics("all_zero_baseline", y_validation, zero_predictions, weights)
    )
    mean_model = DummyRegressor(strategy="mean").fit(
        np.zeros((len(y_train), 1)), y_train
    )
    mean_predictions = mean_model.predict(np.zeros((len(y_validation), 1)))
    metric_rows.append(
        calculate_metrics(
            "training_mean_baseline", y_validation, mean_predictions, weights
        )
    )

    # Timing sensitivity: this version excludes current NOI, occupancy, and
    # delinquency, which would be unavailable in an origination-time forecast.
    origination_linear = build_linear_pipeline(ORIGINATION_NUMERIC_FEATURES)
    origination_linear.fit(x_train, y_train)
    origination_predictions = origination_linear.predict(x_validation)
    metric_rows.append(
        calculate_metrics(
            "linear_origination_only",
            y_validation,
            origination_predictions,
            weights,
        )
    )

    # Required linear regression under the primary current-state interpretation.
    current_linear = build_linear_pipeline(CURRENT_STATE_NUMERIC_FEATURES)
    current_linear.fit(x_train, y_train)
    linear_predictions = current_linear.predict(x_validation)
    metric_rows.append(
        calculate_metrics(
            "linear_current_state", y_validation, linear_predictions, weights
        )
    )

    # Target-policy sensitivity: retain supplied values above one rather than
    # clipping them to the range stated in the assignment.
    raw_linear = build_linear_pipeline(CURRENT_STATE_NUMERIC_FEATURES)
    raw_linear.fit(x_train, raw_y_train)
    raw_predictions = raw_linear.predict(x_validation)
    metric_rows.append(
        calculate_metrics(
            "linear_current_state_raw_sensitivity",
            raw_y_validation,
            raw_predictions,
            weights,
            target_policy="raw supplied severity",
        )
    )

    # Alternative model: separately estimate whether a loss occurs and how large
    # it is, then multiply the two components into expected severity.
    hurdle_model = HurdleSeverityModel(seed).fit(x_train, y_train)
    positive_probability, conditional_magnitude = hurdle_model.predict_components(
        x_validation
    )
    hurdle_predictions = positive_probability * conditional_magnitude
    metric_rows.append(
        calculate_metrics(
            "hurdle_current_state", y_validation, hurdle_predictions, weights
        )
    )

    # Evaluate the two hurdle components as well as their combined prediction.
    has_loss = y_validation.gt(0)
    hurdle_components = {
        "roc_auc": roc_auc_score(has_loss, positive_probability),
        "average_precision": average_precision_score(has_loss, positive_probability),
        "brier_score": brier_score_loss(has_loss, positive_probability),
        "positive_magnitude_mae": mean_absolute_error(
            y_validation.loc[has_loss],
            conditional_magnitude[has_loss.to_numpy()],
        ),
    }
    return {
        "metrics": pd.DataFrame(metric_rows),
        "y_validation": y_validation,
        "linear_predictions": linear_predictions,
        "hurdle_predictions": hurdle_predictions,
        "hurdle_components": hurdle_components,
    }


# -----------------------------------------------------------------------------
# Repeated grouped cross-validation
# -----------------------------------------------------------------------------

def repeated_group_cv(
    features: pd.DataFrame,
    target: pd.Series,
    development: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare linear and hurdle models on 15 identical property-grouped folds."""
    groups = development["propname"]
    rows: list[dict[str, float | int | str]] = []
    for repeat in range(1, CV_REPEATS + 1):
        splitter = GroupKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=seed + repeat - 1,
        )
        for fold, (train_index, validation_index) in enumerate(
            splitter.split(features, target, groups=groups), start=1
        ):
            require(
                set(groups.iloc[train_index]).isdisjoint(groups.iloc[validation_index]),
                "A property crosses a CV fold boundary",
            )
            x_train = features.iloc[train_index]
            x_validation = features.iloc[validation_index]
            y_train = target.iloc[train_index]
            y_validation = target.iloc[validation_index]
            weights = development.iloc[validation_index]["loan_size_mm"]

            linear_model = build_linear_pipeline(CURRENT_STATE_NUMERIC_FEATURES)
            linear_model.fit(x_train, y_train)
            hurdle_model = HurdleSeverityModel(seed + repeat - 1).fit(x_train, y_train)
            predictions = {
                "linear_current_state": linear_model.predict(x_validation),
                "hurdle_current_state": hurdle_model.predict(x_validation),
            }
            for model_name, model_predictions in predictions.items():
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        **calculate_metrics(
                            model_name,
                            y_validation,
                            model_predictions,
                            weights,
                        ),
                    }
                )

    # Summarize performance across folds without treating repeated folds as fully
    # independent statistical samples.
    fold_metrics = pd.DataFrame(rows)
    summary = (
        fold_metrics.groupby(["model", "target_policy"], as_index=False)
        .agg(
            validation_folds=("mse", "size"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            negative_prediction_share_mean=(
                "pct_predictions_below_zero",
                "mean",
            ),
        )
        .sort_values("mse_mean")
        .reset_index(drop=True)
    )

    # Both models use identical folds, so calculate improvements fold by fold.
    paired_mse = fold_metrics.pivot(
        index=["repeat", "fold"], columns="model", values="mse"
    )
    paired_mae = fold_metrics.pivot(
        index=["repeat", "fold"], columns="model", values="mae"
    )
    linear_mse = paired_mse["linear_current_state"]
    hurdle_mse = paired_mse["hurdle_current_state"]
    linear_mae = paired_mae["linear_current_state"]
    hurdle_mae = paired_mae["hurdle_current_state"]
    fold_mse_reduction = (linear_mse - hurdle_mse) / linear_mse
    fold_mae_reduction = (linear_mae - hurdle_mae) / linear_mae
    comparison = pd.DataFrame(
        [
            {
                "target_policy": "severity clipped to [0, 1]",
                "repeats": CV_REPEATS,
                "folds_per_repeat": CV_FOLDS,
                "paired_folds": len(paired_mse),
                "hurdle_mse_wins": int(hurdle_mse.lt(linear_mse).sum()),
                "hurdle_mae_wins": int(hurdle_mae.lt(linear_mae).sum()),
                "linear_mse_mean": float(linear_mse.mean()),
                "hurdle_mse_mean": float(hurdle_mse.mean()),
                "mse_reduction_from_mean_errors": float(
                    1 - hurdle_mse.mean() / linear_mse.mean()
                ),
                "fold_mse_reduction_mean": float(fold_mse_reduction.mean()),
                "fold_mse_reduction_std": float(fold_mse_reduction.std(ddof=1)),
                "fold_mse_reduction_min": float(fold_mse_reduction.min()),
                "fold_mse_reduction_max": float(fold_mse_reduction.max()),
                "linear_mae_mean": float(linear_mae.mean()),
                "hurdle_mae_mean": float(hurdle_mae.mean()),
                "mae_reduction_from_mean_errors": float(
                    1 - hurdle_mae.mean() / linear_mae.mean()
                ),
                "fold_mae_reduction_mean": float(fold_mae_reduction.mean()),
                "fold_mae_reduction_std": float(fold_mae_reduction.std(ddof=1)),
            }
        ]
    )
    return summary, comparison


# -----------------------------------------------------------------------------
# Figures, saved predictions, and console reporting
# -----------------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true: pd.Series,
    predictions: np.ndarray,
    metrics: pd.Series,
    output_path: Path,
    *,
    model_label: str,
    title: str,
    color: str,
    axis_limits: tuple[float, float],
) -> None:
    """Save the required true-versus-predicted severity scatter plot."""
    lower, upper = axis_limits
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    ax.scatter(
        y_true,
        predictions,
        s=14,
        alpha=0.22,
        color=color,
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
    ax.set(xlim=(lower, upper), ylim=(lower, upper))
    ax.set_xlabel("True severity")
    ax.set_ylabel(f"{model_label} predicted severity")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(color="#CBD5E1", alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    metric_text = (
        f"Grouped holdout (n={len(y_true):,})\n"
        f"R² = {metrics['r2']:.3f}\n"
        f"MSE = {metrics['mse']:.4f}\n"
        f"MAE = {metrics['mae']:.4f}\n"
        f"Predictions < 0: {metrics['pct_predictions_below_zero']:.1%}\n"
        f"Predictions > 1: {metrics['pct_predictions_above_one']:.1%}"
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


def score_predictions(
    selected_model: str,
    features: pd.DataFrame,
    target: pd.Series,
    scoring_features: pd.DataFrame,
    scoring_source: pd.DataFrame,
    output_path: Path,
    seed: int,
) -> np.ndarray:
    """Refit the chosen model on all labels and write a scored source-order copy."""
    if selected_model == "hurdle_current_state":
        model = HurdleSeverityModel(seed).fit(features, target)
    elif selected_model == "linear_current_state":
        model = build_linear_pipeline(CURRENT_STATE_NUMERIC_FEATURES)
        model.fit(features, target)
    else:
        raise ValueError(f"Unknown scoring model: {selected_model}")

    predictions = model.predict(scoring_features)

    # Guard the scoring contract before writing anything to disk.
    require(len(predictions) == len(scoring_source), "Scoring row count changed")
    require(np.isfinite(predictions).all(), "Scoring predictions are not finite")
    require(
        ((predictions >= 0) & (predictions <= 1)).all(),
        "Scoring predictions fall outside [0, 1]",
    )
    output = scoring_source.copy()
    output["severity"] = predictions
    require(output["loan_id"].equals(scoring_source["loan_id"]), "Row order changed")
    output.to_csv(output_path, index=False)
    return predictions


def print_results(
    metrics: pd.DataFrame,
    hurdle_components: dict[str, float],
    cv_summary: pd.DataFrame,
    cv_comparison: pd.DataFrame,
) -> None:
    """Print the required metrics plus concise hurdle and stability diagnostics."""
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
    ]
    printable = metrics[display_columns].copy()
    numeric = printable.select_dtypes(include="number").columns
    printable[numeric] = printable[numeric].round(6)
    print("\nHOLDOUT RESULTS")
    print(printable.to_string(index=False))
    print(
        "\nHurdle components: "
        f"ROC-AUC={hurdle_components['roc_auc']:.4f}, "
        f"average precision={hurdle_components['average_precision']:.4f}, "
        f"Brier={hurdle_components['brier_score']:.4f}, "
        f"positive-magnitude MAE={hurdle_components['positive_magnitude_mae']:.4f}."
    )
    print("\nREPEATED PROPERTY-GROUPED CROSS-VALIDATION")
    print(cv_summary.round(6).to_string(index=False))
    comparison = cv_comparison.iloc[0]
    print(
        f"Hurdle MSE wins: {int(comparison['hurdle_mse_wins'])}/"
        f"{int(comparison['paired_folds'])}; mean MSE reduction: "
        f"{comparison['mse_reduction_from_mean_errors']:.2%}; mean MAE reduction: "
        f"{comparison['mae_reduction_from_mean_errors']:.2%}."
    )


# -----------------------------------------------------------------------------
# End-to-end orchestration
# -----------------------------------------------------------------------------

def main() -> None:
    """Run validation, modeling, diagnostics, artifact creation, and scoring."""
    args = parse_args()
    figure_dir = args.output_dir / "figures"
    metric_dir = args.output_dir / "metrics"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load validated source data and apply the documented target policy.
    print(
        "PART 2: MODELING LOSS\n"
        f"pandas={pd.__version__}; numpy={np.__version__}; "
        f"scikit-learn={sklearn.__version__}; seed={args.seed}\n"
        "Timing assumption: current-state severity estimation."
    )
    development, scoring, scoring_source = load_and_validate(
        args.loan_data, args.property_data, args.prediction_data
    )
    raw_target = development["severity"]
    target = raw_target.clip(0, 1)
    print(
        f"Development: {len(development):,} loans across "
        f"{development['propname'].nunique():,} properties; "
        f"severity is zero for {target.eq(0).mean():.2%}, "
        f"with {raw_target.gt(1).sum():,} supplied values above one. "
        "Primary target policy: clip severity to [0, 1]."
    )

    # 2. Engineer development and scoring features through the same function.
    features = engineer_features(development)
    scoring_features = engineer_features(scoring)
    require(
        features.columns.tolist() == scoring_features.columns.tolist(),
        "Development and scoring feature schemas differ",
    )
    # 3. Reserve a property-grouped holdout for headline metrics and figures.
    train_index, validation_index = make_holdout(
        features, target, development["propname"], args.seed
    )
    print(
        f"Grouped holdout: {len(train_index):,} training loans, "
        f"{len(validation_index):,} validation loans, property overlap=0."
    )

    # 4. Evaluate all baselines and models on the identical holdout rows.
    holdout = evaluate_holdout(
        features,
        target,
        raw_target,
        development,
        train_index,
        validation_index,
        args.seed,
    )
    metrics = holdout["metrics"]
    y_validation = holdout["y_validation"]
    linear_predictions = holdout["linear_predictions"]
    hurdle_predictions = holdout["hurdle_predictions"]
    metrics_by_model = metrics.set_index("model")
    limits = (
        min(-0.05, float(linear_predictions.min()), float(hurdle_predictions.min())),
        max(
            1.05,
            float(y_validation.max()),
            float(linear_predictions.max()),
            float(hurdle_predictions.max()),
        ),
    )
    plot_actual_vs_predicted(
        y_validation,
        linear_predictions,
        metrics_by_model.loc["linear_current_state"],
        figure_dir / "part2_linear_actual_vs_predicted.png",
        model_label="Linear regression",
        title="Linear regression misses the zero mass and produces negative severities",
        color="#2563EB",
        axis_limits=limits,
    )
    plot_actual_vs_predicted(
        y_validation,
        hurdle_predictions,
        metrics_by_model.loc["hurdle_current_state"],
        figure_dir / "part2_hurdle_actual_vs_predicted.png",
        model_label="Hurdle model",
        title="Hurdle model improves fit and keeps predictions within [0, 1]",
        color="#0F766E",
        axis_limits=limits,
    )

    # 5. Check that the linear-versus-hurdle result is stable across grouped folds.
    cv_summary, cv_comparison = repeated_group_cv(
        features, target, development, args.seed
    )
    metrics.to_csv(metric_dir / "part2_metrics.csv", index=False)
    cv_summary.to_csv(metric_dir / "part2_repeated_group_cv_summary.csv", index=False)
    cv_comparison.to_csv(
        metric_dir / "part2_repeated_group_cv_comparison.csv", index=False
    )
    print_results(
        metrics,
        holdout["hurdle_components"],
        cv_summary,
        cv_comparison,
    )

    # 6. Select by mean grouped-CV MSE, refit on every labeled row, and score the
    # unlabeled file without modifying the original predictions.csv.
    selected_model = str(cv_summary.sort_values("mse_mean").iloc[0]["model"])
    predictions = score_predictions(
        selected_model,
        features,
        target,
        scoring_features,
        scoring_source,
        args.output_dir / "predictions_scored.csv",
        args.seed,
    )
    print(
        f"\nSelected by grouped-CV mean MSE: {selected_model}. "
        f"Wrote {len(predictions):,} predictions ranging from "
        f"{predictions.min():.4f} to {predictions.max():.4f}."
    )


if __name__ == "__main__":
    main()
