"""Baseline definitions, candidate hyperparameters, and recursive forecast routines."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import next_feature_row, training_matrices

MODEL_NAMES = ["Naive", "Seasonal Naive", "Extra Trees", "Hist Gradient Boosting", "Ridge Regression"]
AUTO_MODEL_SELECTION = "Automatic (best rolling CV)"
FORECAST_MODEL_CHOICES = [AUTO_MODEL_SELECTION] + MODEL_NAMES
ML_MODEL_NAMES = MODEL_NAMES[2:]

# Small, explicit search grids. Kept intentionally compact because this project has < 100 monthly points.
def parameter_candidates(name: str) -> list[dict]:
    if name in ("Naive", "Seasonal Naive"):
        return [{}]
    if name == "Extra Trees":
        return [
            # Original Gradio baseline: preserve these settings as a benchmark candidate.
            {"n_estimators": 350, "min_samples_leaf": 2, "max_features": 0.9},
            {"n_estimators": 150, "min_samples_leaf": 2, "max_features": 0.9},
            {"n_estimators": 250, "min_samples_leaf": 2, "max_features": 0.9},
            {"n_estimators": 250, "min_samples_leaf": 4, "max_features": 0.9},
        ]
    if name == "Hist Gradient Boosting":
        return [
            {"max_iter": 120, "learning_rate": 0.05, "l2_regularization": 1.0, "max_leaf_nodes": 5},
            {"max_iter": 160, "learning_rate": 0.05, "l2_regularization": 1.0, "max_leaf_nodes": 7},
        ]
    if name == "Ridge Regression":
        return [{"alpha": 1.0}, {"alpha": 10.0}, {"alpha": 50.0}]
    raise ValueError(f"Unknown model: {name}")


def default_parameters(name: str) -> dict:
    candidates = parameter_candidates(name)
    if name == "Naive" or name == "Seasonal Naive":
        return {}
    if name == "Extra Trees":
        # Keep the original app's configuration as the default forecast model.
        return {"n_estimators": 350, "min_samples_leaf": 2, "max_features": 0.9}
    if name == "Hist Gradient Boosting":
        return candidates[1].copy()
    if name == "Ridge Regression":
        return {"alpha": 10.0}
    raise ValueError(f"Unknown model: {name}")


def build_model(name: str, params: dict | None = None, random_state: int = 42) -> Pipeline:
    config = default_parameters(name)
    if params:
        config.update(params)
    if name == "Extra Trees":
        estimator = ExtraTreesRegressor(
            n_estimators=int(config["n_estimators"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            max_features=config["max_features"],
            random_state=random_state,
            # Monthly series are tiny; single-process fitting avoids joblib startup overhead
            # across the rolling-origin candidate grid and is often faster for this dataset.
            n_jobs=1,
        )
        steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", estimator)]
    elif name == "Hist Gradient Boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=int(config["max_iter"]),
            learning_rate=float(config["learning_rate"]),
            l2_regularization=float(config["l2_regularization"]),
            max_leaf_nodes=int(config["max_leaf_nodes"]),
            random_state=random_state,
        )
        steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", estimator)]
    elif name == "Ridge Regression":
        estimator = Ridge(alpha=float(config["alpha"]))
        steps = [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    else:
        raise ValueError(f"Unknown fitted model: {name}")
    return Pipeline(steps)


def _last_known(series: pd.Series) -> float:
    known = series.dropna()
    if known.empty:
        raise ValueError("No observed target values are available to forecast from.")
    return float(known.iloc[-1])


def seasonal_naive_value(history: pd.Series, next_date: pd.Timestamp) -> float:
    year_ago = next_date - pd.DateOffset(years=1)
    if year_ago in history.index and pd.notna(history.loc[year_ago]):
        return float(history.loc[year_ago])
    same_month = history[(history.index.month == next_date.month) & history.notna()]
    if not same_month.empty:
        return float(same_month.iloc[-1])
    return _last_known(history)


def fit_fitted_model(series: pd.Series, model_name: str, model_params: dict | None = None) -> Pipeline:
    X, y = training_matrices(series)
    if len(y) < 18:
        raise ValueError(
            f"Only {len(y)} usable training rows remain after the 12-month feature warm-up; "
            "at least 18 are required. Add more history or review missing data."
        )
    model = build_model(model_name, params=model_params)
    model.fit(X, y)
    return model


def forecast(
    series: pd.Series,
    horizon: int,
    model_name: str,
    nonnegative: bool | None = None,
    model_params: dict | None = None,
) -> tuple[pd.Series, object | None]:
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one month.")
    if series.dropna().size < 24 or len(series) < 30:
        raise ValueError("At least 30 calendar months and 24 observed monthly totals are recommended for forecasting.")
    history = series.astype(float).copy()
    # Preserve negative forecasts if negative actuals are legitimate (for example, credits/returns).
    if nonnegative is None:
        known_values = history.dropna()
        nonnegative = bool(not known_values.empty and known_values.min() >= 0)
    model = None if model_name in ("Naive", "Seasonal Naive") else fit_fitted_model(history, model_name, model_params=model_params)
    results = []
    for _ in range(int(horizon)):
        next_date, row = next_feature_row(history)
        if model_name == "Naive":
            value = _last_known(history)
        elif model_name == "Seasonal Naive":
            value = seasonal_naive_value(history, next_date)
        else:
            value = float(model.predict(row)[0])
        if nonnegative:
            value = max(0.0, value)
        results.append((next_date, value))
        history.loc[next_date] = value
    output = pd.Series({date: value for date, value in results}, name="Forecast", dtype=float)
    output.index.name = "Date"
    return output, model
