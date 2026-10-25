"""Calendar-safe lag and rolling features for monthly series."""
from __future__ import annotations
import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 6, 12)


def make_feature_frame(series: pd.Series) -> pd.DataFrame:
    """Create past-only features, retaining NaNs for downstream training imputation."""
    y = series.astype(float)
    frame = pd.DataFrame(index=y.index)
    for lag in LAGS:
        frame[f"lag_{lag}"] = y.shift(lag)

    shifted = y.shift(1)
    frame["rolling_mean_3"] = shifted.rolling(3, min_periods=1).mean()
    frame["rolling_mean_6"] = shifted.rolling(6, min_periods=1).mean()
    frame["rolling_median_3"] = shifted.rolling(3, min_periods=1).median()
    frame["rolling_std_3"] = shifted.rolling(3, min_periods=2).std()
    frame["rolling_std_6"] = shifted.rolling(6, min_periods=2).std()
    frame["month_sin"] = np.sin(2 * np.pi * frame.index.month / 12.0)
    frame["month_cos"] = np.cos(2 * np.pi * frame.index.month / 12.0)
    frame["time_idx"] = np.arange(len(frame), dtype=float)
    return frame.replace([np.inf, -np.inf], np.nan)


def training_matrices(series: pd.Series, warmup_months: int = 12):
    X_all = make_feature_frame(series)
    y = series.astype(float)
    warm = np.arange(len(series)) >= warmup_months
    valid_target = y.notna().to_numpy() & np.isfinite(y.to_numpy(dtype=float, na_value=np.nan))
    mask = warm & valid_target
    X = X_all.iloc[np.flatnonzero(mask)]
    target = y.iloc[np.flatnonzero(mask)]
    return X, target


def next_feature_row(history: pd.Series) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Build feature row for the month after history's final calendar timestamp."""
    if len(history) == 0:
        raise ValueError("Cannot forecast an empty series.")
    next_date = pd.Timestamp(history.index[-1]) + pd.offsets.MonthBegin(1)
    extended = pd.concat([history.astype(float), pd.Series([np.nan], index=[next_date], name=history.name)])
    return next_date, make_feature_frame(extended).iloc[[-1]]
