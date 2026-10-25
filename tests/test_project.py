import numpy as np
import pandas as pd
import pytest

from data_pipeline import build_monthly_series, numeric_columns, read_csv
from evaluation import calculate_metrics, fixed_holdout_backtest, rolling_origin_evaluation
from features import make_feature_frame
from models import MODEL_NAMES, default_parameters, parameter_candidates, forecast


def synthetic_series(n=72, seed=17):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="MS")
    values = 1000 + 70 * np.arange(n) + 180 * np.sin(2 * np.pi * idx.month / 12) + rng.normal(0, 40, n)
    return pd.Series(np.maximum(values, 0), index=idx, name="quantity", dtype=float)


def test_monthly_aggregation_and_keep_missing():
    frame = pd.DataFrame({
        "date": ["2024-01-03", "2024-01-22", "2024-03-11", "bad-date"],
        "quantity": [2, 3, 8, 6],
    })
    series, meta = build_monthly_series(frame, "date", "quantity", "Keep missing")
    assert len(series) == 3
    assert series.loc[pd.Timestamp("2024-01-01")] == 5
    assert np.isnan(series.loc[pd.Timestamp("2024-02-01")])
    assert series.loc[pd.Timestamp("2024-03-01")] == 8
    assert meta["invalid_date_rows"] == 1
    assert meta["missing_months"] == ["2024-02"]


def test_semicolon_csv_is_detected(tmp_path):
    path = tmp_path / "semicolon.csv"
    path.write_text("date;quantity\n2024-01-01;2\n2024-02-01;3\n", encoding="utf-8")
    frame = read_csv(path)
    assert list(frame.columns) == ["date", "quantity"]
    assert frame.shape == (2, 2)


def test_dayfirst_dates_parse_when_ambiguous():
    frame = pd.DataFrame({
        "date": ["02/01/2024", "02/03/2024"],
        "quantity": [2, 3],
    })
    series, _ = build_monthly_series(frame, "date", "quantity", dayfirst=True)
    assert series.index[0] == pd.Timestamp("2024-01-01")
    assert series.index[-1] == pd.Timestamp("2024-03-01")
    assert np.isnan(series.loc[pd.Timestamp("2024-02-01")])


def test_monthly_aggregation_zero_policy_is_explicit():
    frame = pd.DataFrame({"date": ["2024-01-03", "2024-03-11"], "quantity": [2, 8]})
    series, _ = build_monthly_series(frame, "date", "quantity", "Assume zero activity")
    assert series.loc[pd.Timestamp("2024-02-01")] == 0.0


def test_numeric_decimal_comma_and_numeric_columns():
    frame = pd.DataFrame({"date": ["2024-01-01", "2024-02-01", "2024-03-01"], "value": ["1,5", "2,5", "3,5"]})
    assert "value" in numeric_columns(frame)
    series, _ = build_monthly_series(frame, "date", "value")
    assert series.iloc[0] == 1.5


def test_features_are_past_only_and_calendar_aligned():
    idx = pd.date_range("2020-01-01", periods=20, freq="MS")
    s = pd.Series(np.arange(20, dtype=float), index=idx)
    f = make_feature_frame(s)
    assert f.loc[idx[13], "lag_1"] == 12.0
    assert f.loc[idx[13], "lag_12"] == 1.0
    assert f.loc[idx[13], "rolling_mean_3"] == np.mean([10.0, 11.0, 12.0])


def test_metrics_handle_zero_actuals_and_report_bias():
    metrics = calculate_metrics([0, 10, 20], [5, 12, 18])
    assert np.isfinite(metrics["WAPE (%)"])
    assert metrics["Points"] == 3
    assert np.isfinite(metrics["MAPE (%)"])
    assert metrics["Bias"] > 0
    all_zero = calculate_metrics([0, 0], [0, 0])
    assert np.isnan(all_zero["WAPE (%)"])


def test_forecast_has_expected_dates_and_nonnegative_predictions():
    s = synthetic_series()
    s.iloc[28] = np.nan  # unknown month stays missing; model must handle missing lag values
    out, _ = forecast(s, 4, "Extra Trees")
    assert len(out) == 4
    assert out.index[0] == s.index[-1] + pd.offsets.MonthBegin(1)
    assert np.isfinite(out.to_numpy()).all()
    assert (out >= 0).all()


def test_negative_targets_are_not_silently_clipped():
    idx = pd.date_range("2018-01-01", periods=48, freq="MS")
    s = pd.Series(-100.0 - np.arange(48, dtype=float), index=idx, name="net_amount")
    out, _ = forecast(s, 2, "Naive")
    assert (out < 0).all()


def test_holdout_scores_only_observed_actuals():
    s = synthetic_series()
    s.iloc[-2] = np.nan
    metrics, detail = fixed_holdout_backtest(s, "Naive", test_months=6)
    assert len(detail) == 6
    assert metrics["Points"] == 5
    assert detail["Date"].is_monotonic_increasing


def test_rolling_origin_evaluation_includes_baselines_and_horizon():
    summary, detail = rolling_origin_evaluation(
        synthetic_series(), model_names=["Naive", "Seasonal Naive", "Ridge Regression"], horizon=3, max_origins=3
    )
    assert set(summary["Model"]) == {"Naive", "Seasonal Naive", "Ridge Regression"}
    assert (summary["Horizon (months)"] == 3).all()
    assert detail["Origin"].nunique() == 3
    # Ridge Regression has three candidate alphas, so its tuning grid adds two extra configurations.
    assert len(detail) == 3 * 3 * 5
    assert summary["Parameters"].nunique() >= 3


def test_bad_column_fails_clearly():
    frame = pd.DataFrame({"date": ["2024-01-01"], "quantity": [1]})
    with pytest.raises(ValueError, match="Date column"):
        build_monthly_series(frame, "missing-date", "quantity")


def test_full_gradio_callback_with_synthetic_csv(tmp_path):
    import os
    from app import analyze
    from models import AUTO_MODEL_SELECTION

    series = synthetic_series(n=72, seed=29)
    frame = pd.DataFrame({"date": series.index.strftime("%Y-%m-%d"), "value": series.values})
    csv_path = tmp_path / "purchasing.csv"
    frame.to_csv(csv_path, index=False)

    outputs = analyze(
        str(csv_path), "date", "value", "Keep missing", False,
        AUTO_MODEL_SELECTION, 3, 6, 3,
    )
    assert len(outputs) == 12
    assert len(outputs[3]) == 3  # forecast table
    assert len(outputs[4]) == 6  # independent holdout window
    assert len(outputs[5]) == 5  # model families
    assert len(outputs[6]) == 5  # best config per family
    assert len(outputs[7]) == 11  # all compact-grid candidates, including original Extra Trees settings
    assert "Model used for forecast" in outputs[0]
    paths = outputs[8:]
    try:
        assert all(os.path.isfile(path) for path in paths)
    finally:
        for path in paths:
            if os.path.isfile(path):
                os.remove(path)


def test_original_extra_trees_baseline_is_restored_as_default():
    expected = {"n_estimators": 350, "min_samples_leaf": 2, "max_features": 0.9}
    assert default_parameters("Extra Trees") == expected
    assert expected in parameter_candidates("Extra Trees")
