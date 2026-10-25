"""Chronological holdout, rolling-origin tuning, and forecasting metrics."""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import next_feature_row
from models import (
    MODEL_NAMES, fit_fitted_model, forecast, _last_known,
    parameter_candidates, seasonal_naive_value,
)


def calculate_metrics(actual, predicted) -> dict:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    valid = np.isfinite(a) & np.isfinite(p)
    a, p = a[valid], p[valid]
    if len(a) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "WAPE (%)": np.nan, "MAPE (%)": np.nan,
                "Bias": np.nan, "Bias (%)": np.nan, "Points": 0}
    errors = p - a
    denominator = float(np.sum(np.abs(a)))
    nonzero = np.abs(a) > 1e-12
    wape = float(np.sum(np.abs(errors)) / denominator * 100.0) if denominator > 0 else np.nan
    mape = float(np.mean(np.abs(errors[nonzero] / a[nonzero])) * 100.0) if nonzero.any() else np.nan
    bias = float(np.mean(errors))
    bias_pct = float(np.sum(errors) / denominator * 100.0) if denominator > 0 else np.nan
    return {
        "MAE": float(mean_absolute_error(a, p)),
        "RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "WAPE (%)": wape,
        "MAPE (%)": mape,
        "Bias": bias,
        "Bias (%)": bias_pct,
        "Points": int(len(a)),
    }


def _predict_next(history: pd.Series, model_name: str, model=None, nonnegative=None) -> float:
    next_date, row = next_feature_row(history)
    if model_name == "Naive":
        value = _last_known(history)
    elif model_name == "Seasonal Naive":
        value = seasonal_naive_value(history, next_date)
    else:
        value = float(model.predict(row)[0])
    return max(0.0, value) if nonnegative else value


def fixed_holdout_backtest(
    series: pd.Series,
    model_name: str,
    test_months: int = 6,
    model_params: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Fixed-origin estimator with rolling one-step predictions on a chronological test window.

    The estimator is trained on months before the holdout; actuals become known one month at a time.
    Missing actual targets are not scored and remain unknown in history.
    """
    if len(series) <= test_months + 30:
        raise ValueError("Not enough history for this holdout size; use a shorter holdout or more data.")
    split = len(series) - int(test_months)
    history = series.iloc[:split].astype(float).copy()
    test = series.iloc[split:]
    known_train = history.dropna()
    nonnegative = bool(not known_train.empty and known_train.min() >= 0)
    model = None if model_name in ("Naive", "Seasonal Naive") else fit_fitted_model(history, model_name, model_params=model_params)
    rows = []
    for date, actual_value in test.items():
        prediction = _predict_next(history, model_name, model=model, nonnegative=nonnegative)
        rows.append({"Date": date, "Actual": actual_value, "Predicted": prediction,
                     "Error": prediction - actual_value if pd.notna(actual_value) else np.nan,
                     "Absolute error": abs(prediction - actual_value) if pd.notna(actual_value) else np.nan})
        history.loc[date] = float(actual_value) if pd.notna(actual_value) else np.nan
    result = pd.DataFrame(rows)
    metrics = calculate_metrics(result["Actual"], result["Predicted"])
    metrics["Test months"] = int(test_months)
    return metrics, result


def _origins(n_months: int, horizon: int, max_origins: int, minimum_train_months: int = 36) -> list[int]:
    last_origin = n_months - horizon
    first_origin = max(minimum_train_months, n_months // 2)
    if last_origin < first_origin:
        raise ValueError("Not enough calendar history for rolling-origin evaluation at this horizon. Reduce the holdout/horizon or add history.")
    count = min(max_origins, last_origin - first_origin + 1)
    return sorted(set(int(x) for x in np.linspace(first_origin, last_origin, num=count)))


def rolling_origin_evaluation(
    series: pd.Series,
    model_names: list[str] | None = None,
    horizon: int = 3,
    max_origins: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune a small, deterministic candidate grid on rolling-origin validation data.

    Each candidate is evaluated on identical origins. Every origin trains only on its past prefix
    and recursively predicts the requested horizon without observing targets inside that window.
    The caller is responsible for passing only the development period, not the final test window.
    """
    names = model_names or MODEL_NAMES
    origins = _origins(len(series), int(horizon), int(max_origins))
    candidates: list[tuple[str, dict, str]] = []
    for name in names:
        for params in parameter_candidates(name):
            params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
            candidates.append((name, params, params_json))

    all_rows: list[dict] = []
    for origin_number, origin in enumerate(origins, start=1):
        train = series.iloc[:origin].astype(float)
        actual_window = series.iloc[origin:origin + horizon]
        for model_name, model_params, params_json in candidates:
            started = time.perf_counter()
            predicted, _ = forecast(train, int(horizon), model_name, nonnegative=None, model_params=model_params)
            elapsed = time.perf_counter() - started
            for k, date in enumerate(actual_window.index):
                actual = actual_window.iloc[k]
                pred = float(predicted.iloc[k])
                all_rows.append({
                    "Origin": origin_number,
                    "Origin date": train.index[-1],
                    "Date": date,
                    "Horizon step": k + 1,
                    "Model": model_name,
                    "Parameters": params_json,
                    "Actual": actual,
                    "Predicted": pred,
                    "Absolute error": abs(pred - actual) if pd.notna(actual) else np.nan,
                    "Fit + forecast seconds": elapsed,
                })
    details = pd.DataFrame(all_rows)
    summary_rows = []
    for (model_name, params_json), group in details.groupby(["Model", "Parameters"], sort=False):
        metrics = calculate_metrics(group["Actual"], group["Predicted"])
        fold_wapes = []
        for _, fold in group.groupby("Origin"):
            fold_metric = calculate_metrics(fold["Actual"], fold["Predicted"])
            if np.isfinite(fold_metric["WAPE (%)"]):
                fold_wapes.append(fold_metric["WAPE (%)"])
        summary_rows.append({
            "Model": model_name,
            "Parameters": params_json,
            **metrics,
            "Origins": int(group["Origin"].nunique()),
            "Mean fold WAPE (%)": float(np.mean(fold_wapes)) if fold_wapes else np.nan,
            "Std fold WAPE (%)": float(np.std(fold_wapes, ddof=1)) if len(fold_wapes) > 1 else np.nan,
            "Mean fit + forecast (s)": float(group.groupby("Origin")["Fit + forecast seconds"].first().mean()),
            "Horizon (months)": int(horizon),
        })
    summary = pd.DataFrame(summary_rows).sort_values("WAPE (%)", na_position="last").reset_index(drop=True)
    return summary, details


def compare_fixed_holdout(
    series: pd.Series,
    test_months: int = 6,
    model_names=None,
    model_params_by_name: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compare one validation-selected configuration per model family on a separate holdout."""
    names = model_names or MODEL_NAMES
    params_by_name = model_params_by_name or {}
    summaries, details = [], {}
    for name in names:
        params = params_by_name.get(name)
        metrics, result = fixed_holdout_backtest(series, name, test_months, model_params=params)
        summaries.append({"Model": name, "Parameters": json.dumps(params or {}, sort_keys=True, separators=(",", ":")), **metrics})
        details[name] = result
    summary = pd.DataFrame(summaries).sort_values("WAPE (%)", na_position="last").reset_index(drop=True)
    return summary, details
