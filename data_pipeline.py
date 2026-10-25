"""CSV loading, monthly aggregation, and data-quality diagnostics."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_DATE_CANDIDATES = ["Date transfert", "Date prévue", "date", "Date"]


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV robustly while preserving the caller's source file."""
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="latin-1")
        encoding = "latin-1"
    # Some exported business reports use semicolons or tabs rather than commas.
    if len(frame.columns) == 1:
        try:
            detected = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
            if len(detected.columns) > 1:
                frame = detected
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
            pass
    if frame.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def date_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if "date" in c.casefold()]


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    found: list[str] = []
    for col in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[col]):
            found.append(col)
        else:
            converted = _numeric_values(frame[col])
            if converted.notna().sum() >= max(3, int(0.75 * frame[col].notna().sum())):
                found.append(col)
    return found


def _numeric_values(values: pd.Series) -> pd.Series:
    """Convert numbers and common decimal-comma strings without guessing for floats."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")

    def convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        raw = str(value).strip().replace("\u00a0", "").replace(" ", "")
        if not raw:
            return np.nan
        if "," in raw and "." in raw:
            # The rightmost punctuation is treated as the decimal separator.
            if raw.rfind(",") > raw.rfind("."):
                raw = raw.replace(".", "").replace(",", ".")
            else:
                raw = raw.replace(",", "")
        elif "," in raw:
            raw = raw.replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            return np.nan

    return values.map(convert).astype(float)


def infer_column(frame: pd.DataFrame, candidates: list[str], kind: str) -> str | None:
    lookup = {str(c).strip().casefold(): c for c in frame.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    if kind == "date":
        cols = date_columns(frame)
        return cols[0] if cols else None
    nums = numeric_columns(frame)
    return nums[0] if nums else None


def build_monthly_series(
    frame: pd.DataFrame,
    date_col: str,
    target_col: str,
    missing_policy: str = "Keep missing",
    dayfirst: bool = False,
) -> tuple[pd.Series, dict]:
    """Aggregate transaction rows into calendar months.

    Missing month targets remain NaN unless the user explicitly chooses zero fill.
    Valid source transactions are not deduplicated because repeated documents/lines may be legitimate.
    """
    if date_col not in frame.columns:
        raise ValueError(f"Date column '{date_col}' was not found.")
    if target_col not in frame.columns:
        raise ValueError(f"Target column '{target_col}' was not found.")

    # ISO year-first dates are unambiguous; parse them separately so a day-first preference
    # does not emit warnings or accidentally reinterpret their components.
    date_text = frame[date_col].astype("string").str.strip()
    sample_dates = date_text.dropna().head(100)
    iso_like_ratio = float(sample_dates.str.match(r"^\d{4}-\d{1,2}-\d{1,2}").mean()) if len(sample_dates) else 0.0
    if iso_like_ratio >= 0.8:
        dates = pd.to_datetime(frame[date_col], errors="coerce", format="ISO8601")
    else:
        dates = pd.to_datetime(frame[date_col], errors="coerce", dayfirst=dayfirst)
    values = _numeric_values(frame[target_col])
    finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    valid = dates.notna().to_numpy() & values.notna().to_numpy() & finite
    usable = pd.DataFrame({
        "Date": dates.loc[valid].dt.to_period("M").dt.to_timestamp(),
        "Value": values.loc[valid].astype(float),
    })
    if usable.empty:
        raise ValueError("No valid date/target records remain. Check the selected columns and date format.")

    # GroupBy sum is the appropriate aggregate for transaction-level quantity/cost data.
    observed_monthly = usable.groupby("Date", sort=True)["Value"].sum().sort_index()
    full_index = pd.date_range(observed_monthly.index.min(), observed_monthly.index.max(), freq="MS")
    absent = full_index.difference(observed_monthly.index)
    monthly = observed_monthly.reindex(full_index).astype(float)

    if missing_policy == "Assume zero activity":
        monthly = monthly.fillna(0.0)
    elif missing_policy != "Keep missing":
        raise ValueError(f"Unknown missing-month policy: {missing_policy}")

    monthly.index.name = "Date"
    monthly.name = target_col
    observed_vals = observed_monthly.to_numpy(dtype=float)
    row_values = usable["Value"].to_numpy(dtype=float)
    q1, q3 = np.quantile(row_values, [0.25, 0.75])
    iqr = float(q3 - q1)
    lower_fence, upper_fence = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
    outlier_count = int(((row_values < lower_fence) | (row_values > upper_fence)).sum())
    meta = {
        "raw_rows": int(len(frame)),
        "valid_rows": int(valid.sum()),
        "invalid_date_rows": int(dates.isna().sum()),
        "invalid_target_rows": int((values.isna() | ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))).sum()),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "observed_months": int(len(observed_monthly)),
        "calendar_months": int(len(monthly)),
        "missing_months_count": int(len(absent)),
        "missing_months": [d.strftime("%Y-%m") for d in absent],
        "date_col": date_col,
        "target_col": target_col,
        "start": monthly.index.min().strftime("%Y-%m"),
        "end": monthly.index.max().strftime("%Y-%m"),
        "missing_policy": missing_policy,
        "target_min": float(np.min(observed_vals)),
        "target_median": float(np.median(observed_vals)),
        "target_max": float(np.max(observed_vals)),
        "target_q1": float(q1),
        "target_q3": float(q3),
        "iqr_lower_fence": lower_fence,
        "iqr_upper_fence": upper_fence,
        "iqr_outlier_rows": outlier_count,
        "negative_transactions": int((usable["Value"] < 0).sum()),
        "duplicate_rows_not_removed": True,
    }
    return monthly, meta
