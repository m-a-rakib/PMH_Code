#!/usr/bin/env python3
"""Reproduce key descriptive, data-quality, scale, and illustrative model results.

This script contains no participant records or audio. It reads the released
XLSX dataset supplied by the user and writes tables to an Excel workbook.

Example:
    python reproduce_results.py --input "PMH Dataset.xlsx" --output "PMH_results.xlsx"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

PHQ_ITEMS = [f"phq{i}" for i in range(1, 10)]
GAD_ITEMS = [f"gad{i}" for i in range(1, 8)]
AUDIO_COLUMNS = ["audio_fs_file", "audio_ct_file", "audio_rd_file"]

PHQ_PREDICTORS = [
    "female", "age_35plus", "sleep_falling_often",
    "high_functional_impact", "financial_stress",
    "low_social_support_talk", "very_slow_speech",
]
GAD_PREDICTORS = [
    "female", "age_35plus", "poor_role_functioning",
    "low_life_satisfaction", "financial_stress",
    "bereavement_stress", "legal_police_stress",
    "sleep_falling_often",
]


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the same column-name normalization used in the analysis notebook."""
    out = data.copy()
    out.columns = (
        out.columns.astype(str).str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.lower()
    )
    return out


def clean_series(series: pd.Series) -> pd.Series:
    out = series.copy()
    if pd.api.types.is_object_dtype(out) or pd.api.types.is_string_dtype(out):
        out = out.astype("string").str.strip()
        out = out.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    return out


def read_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        data = pd.read_excel(path)
    elif suffix == ".csv":
        data = pd.read_csv(path)
    else:
        raise ValueError("Input must be a .xlsx, .xls, or .csv file.")
    return normalize_columns(data)


RESPONSE_SCORES = {
    "not at all": 0,
    "several days": 1,
    "more than half the days": 2,
    "nearly every day": 3,
}


def item_scores(items: pd.DataFrame) -> pd.DataFrame:
    """Return 0-3 item scores.

    The public dataset stores PHQ-9/GAD-7 item responses as text labels
    (for example "Several days"). Labels are converted with the scoring map
    above; values that are already numeric are kept as they are.
    """
    def convert(col: pd.Series) -> pd.Series:
        mapped = col.astype("string").str.strip().str.lower().map(RESPONSE_SCORES)
        return mapped.fillna(pd.to_numeric(col, errors="coerce")).astype(float)
    return items.apply(convert)


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Calculate Cronbach's alpha using complete item rows."""
    x = item_scores(items).dropna()
    k = x.shape[1]
    if k < 2 or len(x) < 2:
        return float("nan")
    item_variances = x.var(axis=0, ddof=1)
    total_variance = x.sum(axis=1).var(ddof=1)
    if not np.isfinite(total_variance) or total_variance <= 0:
        return float("nan")
    return float((k / (k - 1)) * (1 - item_variances.sum() / total_variance))


def score_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["phq_total", "gad_total"]:
        s = pd.to_numeric(data[col], errors="coerce").dropna()
        rows.append({
            "score": col,
            "n": int(s.size),
            "mean": s.mean(),
            "sd": s.std(ddof=1),
            "median": s.median(),
            "q1": s.quantile(0.25),
            "q3": s.quantile(0.75),
            "min": s.min(),
            "max": s.max(),
        })
    return pd.DataFrame(rows)


def missingness_summary(data: pd.DataFrame) -> pd.DataFrame:
    n = len(data)
    return pd.DataFrame({
        "variable": data.columns,
        "n_rows": n,
        "missing_n": data.isna().sum().values,
        "missing_percent": (data.isna().mean().values * 100).round(3),
        "nonmissing_n": data.notna().sum().values,
    }).sort_values(["missing_n", "variable"], ascending=[False, True])


def categorical_frequencies(data: pd.DataFrame) -> pd.DataFrame:
    """Long-format counts/percentages for categorical columns, excluding ID/audio fields."""
    excluded = {"study_id", *AUDIO_COLUMNS}
    rows = []
    for col in data.columns:
        if col in excluded or pd.api.types.is_numeric_dtype(data[col]):
            continue
        s = clean_series(data[col]).dropna()
        if s.empty:
            continue
        counts = s.value_counts(dropna=False)
        denom = len(s)
        for category, count in counts.items():
            rows.append({
                "variable": col,
                "category": str(category),
                "n": int(count),
                "percent_nonmissing": 100 * count / denom,
                "nonmissing_n": denom,
            })
    return pd.DataFrame(rows)


def yes_no_binary(series: pd.Series, yes_value: str = "Yes") -> np.ndarray:
    """Reproduce the original notebook's Yes=1, all other nonmissing=0 rule."""
    s = clean_series(series)
    return np.where(s.isna(), np.nan, np.where(s == yes_value, 1, 0))


def derive_analysis_variables(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    for col in ["phq_total", "gad_total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Item responses are text labels in the public file; convert them to 0-3 scores.
    item_cols = [c for c in [*PHQ_ITEMS, *GAD_ITEMS] if c in df.columns]
    df[item_cols] = item_scores(df[item_cols])

    # The source notebook defines the binary outcomes at a score threshold of 10.
    df["phq_moderate_severe"] = np.where(df["phq_total"] >= 10, 1, 0)
    df["gad_moderate_severe"] = np.where(df["gad_total"] >= 10, 1, 0)

    df["female"] = np.where(clean_series(df["sex"]) == "Female", 1, 0)
    df["age_35plus"] = np.where(
        clean_series(df["age_group"]).isin(["35-44", "45-54", "55-64", ">= 65"]), 1, 0
    )
    df["sleep_falling_often"] = np.where(
        clean_series(df["sleep_diff_falling_asleep"]).isin(["Often", "Almost every day"]), 1, 0
    )
    df["high_functional_impact"] = np.where(
        clean_series(df["functional_impact"]).isin(["Very difficult", "Extremely difficult"]), 1, 0
    )
    df["poor_role_functioning"] = np.where(
        clean_series(df["role_functioning"]).isin(["Very difficult", "Can not do at all"]), 1, 0
    )
    df["low_life_satisfaction"] = np.where(
        clean_series(df["life_satisfaction"]).isin(["Dissatisfied", "Very dissatisfied"]), 1, 0
    )
    df["low_social_support_talk"] = np.where(
        clean_series(df["social_support_talk"]).isin(["Never", "Sometimes"]), 1, 0
    )
    df["financial_stress"] = yes_no_binary(df["stressor_financial"])
    df["bereavement_stress"] = yes_no_binary(df["stressor_bereavement"])
    df["legal_police_stress"] = yes_no_binary(df["stressor_legal_police"])
    df["very_slow_speech"] = yes_no_binary(df["sb_very_slow_speech"])
    return df


def outcome_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col in [("PHQ-9 >= 10", "phq_moderate_severe"), ("GAD-7 >= 10", "gad_moderate_severe")]:
        y = pd.to_numeric(data[col], errors="coerce").dropna().astype(int)
        rows.append({
            "outcome": label,
            "threshold": 10,
            "n": len(y),
            "score_below_10_n": int((y == 0).sum()),
            "score_at_least_10_n": int((y == 1).sum()),
            "score_at_least_10_percent": 100 * float((y == 1).mean()) if len(y) else np.nan,
        })
    return pd.DataFrame(rows)


def fit_logistic(data: pd.DataFrame, outcome: str, predictors: list[str]) -> tuple[pd.DataFrame, object, pd.DataFrame, pd.Series]:
    temp = data[[outcome] + predictors].dropna().copy()
    y = temp[outcome].astype(int)
    X = sm.add_constant(temp[predictors].astype(float), has_constant="add")
    model = sm.Logit(y, X).fit(disp=False, maxiter=200)
    conf = model.conf_int()
    results = pd.DataFrame({
        "predictor": predictors,
        "beta": [model.params[p] for p in predictors],
        "adjusted_OR": [np.exp(model.params[p]) for p in predictors],
        "CI_lower": [np.exp(conf.loc[p, 0]) for p in predictors],
        "CI_upper": [np.exp(conf.loc[p, 1]) for p in predictors],
        "p_value": [model.pvalues[p] for p in predictors],
        "n_used": len(temp),
        "events": int(y.sum()),
        "non_events": int((y == 0).sum()),
    })
    return results, model, X, y


def performance_row(outcome: str, y: pd.Series, probabilities: np.ndarray, evaluation: str) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "outcome": outcome,
        "evaluation": evaluation,
        "n": len(y),
        "events": int((y == 1).sum()),
        "non_events": int((y == 0).sum()),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": accuracy_score(y, predictions),
        "sensitivity": recall_score(y, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": precision_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
        "AUC": roc_auc_score(y, probabilities),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "average_precision": average_precision_score(y, probabilities),
        "Brier_score": brier_score_loss(y, probabilities),
    }


def model_performance(data: pd.DataFrame, outcome: str, predictors: list[str], model, n_splits: int = 5) -> list[dict]:
    temp = data[[outcome] + predictors].dropna().copy()
    y = temp[outcome].astype(int)
    X_sm = sm.add_constant(temp[predictors].astype(float), has_constant="add")
    in_sample_prob = np.asarray(model.predict(X_sm))
    rows = [performance_row(outcome, y, in_sample_prob, "In-sample")]

    # Match the original notebook's CV settings: unscaled LogisticRegression,
    # liblinear solver, shuffled stratified folds, random_state=42, threshold=.5.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    cv_prob = cross_val_predict(clf, temp[predictors].astype(float), y,
                                cv=cv, method="predict_proba")[:, 1]
    rows.append(performance_row(outcome, y, cv_prob, f"{n_splits}-fold CV"))
    return rows


def validate_columns(df: pd.DataFrame) -> None:
    required = [
        "phq_total", "gad_total", *PHQ_ITEMS, *GAD_ITEMS,
        "sex", "age_group", "sleep_diff_falling_asleep", "functional_impact",
        "role_functioning", "life_satisfaction", "social_support_talk",
        "stressor_financial", "stressor_bereavement", "stressor_legal_police",
        "sb_very_slow_speech",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Required columns are missing after name normalization: " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to released CSV or Excel dataset")
    parser.add_argument("--output", default="PMH_reproducibility_results.xlsx", help="Output Excel workbook")
    parser.add_argument("--skip-models", action="store_true", help="Run data-quality and scale summaries only")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    df = read_data(input_path)
    validate_columns(df)
    df = derive_analysis_variables(df)

    # Key data and score checks
    data_overview = pd.DataFrame([{
        "input_file": input_path.name,
        "n_participants": len(df),
        "n_variables_input": len([c for c in df.columns if c not in {
            "phq_moderate_severe", "gad_moderate_severe", "female", "age_35plus",
            "sleep_falling_often", "high_functional_impact", "poor_role_functioning",
            "low_life_satisfaction", "low_social_support_talk", "financial_stress",
            "bereavement_stress", "legal_police_stress", "very_slow_speech"
        }]),
    }])
    miss = missingness_summary(read_data(input_path))
    scores = score_summary(df)
    alpha = pd.DataFrame([
        {"scale": "PHQ-9", "items": len(PHQ_ITEMS), "complete_item_rows": int(df[PHQ_ITEMS].notna().all(axis=1).sum()),
         "cronbach_alpha": cronbach_alpha(df[PHQ_ITEMS])},
        {"scale": "GAD-7", "items": len(GAD_ITEMS), "complete_item_rows": int(df[GAD_ITEMS].notna().all(axis=1).sum()),
         "cronbach_alpha": cronbach_alpha(df[GAD_ITEMS])},
    ])
    rho, rho_p = spearmanr(df["phq_total"], df["gad_total"], nan_policy="omit")
    correlation = pd.DataFrame([{"variable_1": "phq_total", "variable_2": "gad_total",
                                 "spearman_rho": rho, "p_value": rho_p,
                                 "n_complete": int(df[["phq_total", "gad_total"]].dropna().shape[0])}])
    outcome_counts = outcome_summary(df)
    audio_rows = []
    for col in AUDIO_COLUMNS:
        if col in df.columns:
            audio_rows.append({"variable": col, "available_n": int(df[col].notna().sum()),
                               "missing_n": int(df[col].isna().sum()),
                               "available_percent": 100 * float(df[col].notna().mean())})
    audio_availability = pd.DataFrame(audio_rows)
    frequencies = categorical_frequencies(read_data(input_path))

    results: dict[str, pd.DataFrame] = {
        "data_overview": data_overview,
        "missingness": miss,
        "score_summary": scores,
        "scale_reliability": alpha,
        "PHQ_GAD_correlation": correlation,
        "outcome_counts": outcome_counts,
        "audio_availability": audio_availability,
        "categorical_frequencies": frequencies,
    }

    if not args.skip_models:
        phq_results, phq_model, _, _ = fit_logistic(df, "phq_moderate_severe", PHQ_PREDICTORS)
        gad_results, gad_model, _, _ = fit_logistic(df, "gad_moderate_severe", GAD_PREDICTORS)
        perf_rows = model_performance(df, "phq_moderate_severe", PHQ_PREDICTORS, phq_model)
        perf_rows += model_performance(df, "gad_moderate_severe", GAD_PREDICTORS, gad_model)
        results["PHQ_logistic_OR"] = phq_results
        results["GAD_logistic_OR"] = gad_results
        results["model_performance"] = pd.DataFrame(perf_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet, table in results.items():
            table.to_excel(writer, sheet_name=sheet[:31], index=False)

    print(f"Input dataset: {input_path}")
    print(f"Rows: {len(df)}; input variables: {len(read_data(input_path).columns)}")
    print(f"Results saved to: {output_path}")
    print("Note: this script does not include participant records or audio in the output workbook.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
