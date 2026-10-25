"""Reproduce baseline, compact hyperparameter search, holdout, and rolling-origin results."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline import read_csv, build_monthly_series
from evaluation import compare_fixed_holdout, fixed_holdout_backtest, rolling_origin_evaluation
from models import MODEL_NAMES, default_parameters


def original_app_features(series):
    frame = pd.DataFrame({"y": series.astype(float)})
    for lag in (1, 2, 3, 6, 12):
        frame[f"lag_{lag}"] = frame["y"].shift(lag)
    frame["rolling_mean_3"] = frame["y"].shift(1).rolling(3).mean()
    frame["rolling_mean_6"] = frame["y"].shift(1).rolling(6).mean()
    frame["rolling_std_3"] = frame["y"].shift(1).rolling(3).std()
    frame["month_sin"] = np.sin(2 * np.pi * frame.index.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame.index.month / 12)
    frame["time_idx"] = np.arange(len(frame))
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def original_baseline(series, test_months=6):
    feat = original_app_features(series)
    split = feat.index[-test_months]
    train_series = series.loc[series.index < split]
    train_feat = original_app_features(train_series)
    model = ExtraTreesRegressor(
        n_estimators=350, min_samples_leaf=2, max_features=0.9, random_state=42, n_jobs=1
    )
    X_train, y_train = train_feat.drop(columns="y"), train_feat["y"]
    model.fit(X_train, y_train)
    history = train_series.copy()
    rows = []
    test_series = series.loc[series.index >= split]
    for date, actual in test_series.items():
        f = original_app_features(history).drop(columns="y").iloc[[-1]]
        pred = max(0.0, float(model.predict(f)[0]))
        rows.append({"Date": date, "Actual": actual, "Predicted": pred})
        history.loc[date] = actual
    res = pd.DataFrame(rows)
    a = res["Actual"].to_numpy(dtype=float); p = res["Predicted"].to_numpy(dtype=float)
    err = p-a; denom = np.abs(a).sum()
    return {
        "Model": "Original app Extra Trees (350 trees)",
        "MAE": mean_absolute_error(a,p),
        "RMSE": np.sqrt(mean_squared_error(a,p)),
        "WAPE (%)": np.abs(err).sum()/denom*100 if denom else np.nan,
        "MAPE (%)": np.mean(np.abs(err)/np.maximum(np.abs(a),1e-9))*100,
        "Bias": err.mean(), "Bias (%)": err.sum()/denom*100 if denom else np.nan,
        "Points": len(a), "Test months": test_months,
    }


def best_params_by_family(summary):
    valid = summary.loc[np.isfinite(pd.to_numeric(summary["WAPE (%)"], errors="coerce"))]
    best = valid.sort_values("WAPE (%)").groupby("Model", sort=False, as_index=False).first()
    mapping = {row["Model"]: json.loads(row["Parameters"]) for _, row in best.iterrows()}
    return best, mapping


def show_number(v, digits=2):
    return "N/A" if not np.isfinite(float(v)) else f"{float(v):,.{digits}f}"


def main():
    csv = ROOT / "cleaned_achat_data.csv"
    raw = read_csv(csv)
    s_keep, meta = build_monthly_series(raw, "Date transfert", "Quantité", "Keep missing")
    s_zero, _ = build_monthly_series(raw, "Date transfert", "Quantité", "Assume zero activity")
    original = original_baseline(s_zero, 6)

    candidate_summaries = []
    best_family_rows = []
    best_params_at_3m: dict[str, dict] = {}
    for policy, full_series in [("Assume zero activity", s_zero), ("Keep missing", s_keep)]:
        development = full_series.iloc[:-6]  # final six months are never used to tune/select
        for horizon in (1, 3, 6):
            print(f"[benchmark] {policy}: tuning horizon={horizon} using development months only", flush=True)
            tuned, _ = rolling_origin_evaluation(development, MODEL_NAMES, horizon=horizon, max_origins=5)
            print(f"[benchmark] {policy}: horizon={horizon} done ({len(tuned)} candidate configurations)", flush=True)
            tuned.insert(0, "Missing policy", policy)
            candidate_summaries.append(tuned)
            best_family, params_by_model = best_params_by_family(tuned)
            best_family_rows.append(best_family)
            if horizon == 3:
                best_params_at_3m[policy] = params_by_model

    all_candidates = pd.concat(candidate_summaries, ignore_index=True)
    best_cv = pd.concat(best_family_rows, ignore_index=True)

    fixed_rows = []
    for policy, series in [("Assume zero activity", s_zero), ("Keep missing", s_keep)]:
        fixed, _ = compare_fixed_holdout(
            series, test_months=6, model_names=MODEL_NAMES,
            model_params_by_name=best_params_at_3m[policy],
        )
        fixed.insert(0, "Missing policy", policy)
        fixed_rows.append(fixed)
    fixed_all = pd.concat(fixed_rows, ignore_index=True)

    # Evaluate the exact default used by the upgraded Gradio app, not only the CV-tuned
    # Extra Trees candidate. This makes the displayed app default reproducible in the report.
    restored_default_rows = []
    restored_params = default_parameters("Extra Trees")
    for policy, series in [("Assume zero activity", s_zero), ("Keep missing", s_keep)]:
        metrics, _ = fixed_holdout_backtest(
            series, "Extra Trees", test_months=6, model_params=restored_params,
        )
        restored_default_rows.append({
            "Missing policy": policy,
            "Model": "Extra Trees (restored original settings)",
            "Parameters": json.dumps(restored_params, sort_keys=True, separators=(",", ":")),
            **metrics,
        })
    fixed_all = pd.concat([fixed_all, pd.DataFrame(restored_default_rows)], ignore_index=True)
    original_df = pd.DataFrame([original])

    fixed_all.to_csv(ROOT / "model_comparison_holdout.csv", index=False)
    best_cv.to_csv(ROOT / "model_comparison_rolling_origin.csv", index=False)
    all_candidates.to_csv(ROOT / "model_tuning_candidates.csv", index=False)
    original_df.to_csv(ROOT / "original_baseline.csv", index=False)

    lines = [
        "# Model evaluation report — bundled purchasing data", "",
        "## Dataset audit", "",
        f"- Raw rows: {meta['raw_rows']:,}",
        f"- Valid date + target rows: {meta['valid_rows']:,}",
        f"- Exact duplicate transaction rows (not removed): {meta['exact_duplicate_rows']:,}",
        f"- Target: `Quantité`; date: `Date transfert`",
        f"- Calendar span: {meta['start']} to {meta['end']} ({meta['calendar_months']} months)",
        f"- Source months absent: {meta['missing_months_count']} ({', '.join(meta['missing_months'])})",
        f"- Transaction target IQR outlier flags: {meta['iqr_outlier_rows']:,}; this is a review heuristic, not an automatic deletion rule.", "",
        "## Original app baseline", "",
        "The former app's Extra Trees configuration (350 trees, original feature set) was re-run on the final six months with missing months explicitly filled with zero. It reproduces the prior reported baseline. This holdout is the same final six-month window used below.", "",
        "| Metric | Original app baseline |", "|---|---:|",
        f"| MAE | {original['MAE']:,.2f} |",
        f"| RMSE | {original['RMSE']:,.2f} |",
        f"| WAPE | {original['WAPE (%)']:.4f}% |",
        f"| MAPE | {original['MAPE (%)']:.4f}% |", "",
        "## Hyperparameter tuning", "",
        "A compact, deterministic grid is evaluated on identical rolling-origin folds in the development period, which ends before the final six-month test window. The search is intentionally small for this 84-month dataset. The best configuration per model family is then evaluated on the final holdout; the holdout is not used to choose parameters. The Gradio forecast control now defaults to the original Extra Trees settings (`n_estimators=350`, `min_samples_leaf=2`, `max_features=0.9`), while automatic rolling-origin selection remains optional.", "",
        "| Model family | Candidate configurations |", "|---|---|",
        "| Naive | Fixed baseline |",
        "| Seasonal Naive | Fixed baseline, with fallback if the same month last year is missing |",
        "| Extra Trees | 150/250/350 trees; `min_samples_leaf` 2/4; includes original 350-tree baseline |",
        "| Hist Gradient Boosting | 120/160 iterations; `max_leaf_nodes` 5/7 |",
        "| Ridge Regression | alpha 1 / 10 / 50 |", "",
        "## Final six-month chronological holdout", "",
        "All model families are evaluated on the same final six calendar months (July–December 2025). For each family, its hyperparameters were selected on development-period rolling-origin folds at a three-month horizon. These holdout scores are a separate final diagnostic, not a tuning target.", "",
    ]
    for policy in ["Assume zero activity", "Keep missing"]:
        lines += [f"### Missing policy: {policy}", "", "| Model | Selected parameters | WAPE | MAE | RMSE | MAPE | Bias |", "|---|---|---:|---:|---:|---:|---:|"]
        group = fixed_all[fixed_all["Missing policy"] == policy].sort_values("WAPE (%)")
        for _, row in group.iterrows():
            lines.append(
                f"| {row['Model']} | `{row['Parameters']}` | {row['WAPE (%)']:.2f}% | {row['MAE']:,.2f} | "
                f"{row['RMSE']:,.2f} | {row['MAPE (%)']:.2f}% | {row['Bias']:,.2f} |"
            )
        lines.append("")
    lines += [
        "## Rolling-origin validation results", "",
        "Each fold trains on an historical prefix and recursively forecasts 1, 3, or 6 months without receiving actual values within the forecast window. The final six months are excluded from every validation fold used for model selection. Metrics aggregate all scoreable prediction points; the standard deviation of fold WAPE indicates instability across origins.", "",
        "| Missing policy | Horizon | Model | Selected parameters | WAPE | MAE | RMSE | MAPE | Mean fold WAPE | Std fold WAPE | Origins |", "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in best_cv.sort_values(["Missing policy", "Horizon (months)", "WAPE (%)"]).iterrows():
        lines.append(
            f"| {row['Missing policy']} | {int(row['Horizon (months)'])} | {row['Model']} | `{row['Parameters']}` | "
            f"{row['WAPE (%)']:.2f}% | {row['MAE']:,.2f} | {row['RMSE']:,.2f} | {row['MAPE (%)']:.2f}% | "
            f"{row['Mean fold WAPE (%)']:.2f}% | {row['Std fold WAPE (%)']:.2f}% | {int(row['Origins'])} |"
        )
    lines += [
        "", "## Interpretation", "",
        "- The original app's WAPE is a baseline for one recent six-month rolling one-step test, not a universal forecast accuracy percentage.",
        f"- On the zero-fill final holdout, the current pipeline's validation-selected Extra Trees configuration has {fixed_all.loc[(fixed_all['Missing policy'] == 'Assume zero activity') & (fixed_all['Model'] == 'Extra Trees'), 'WAPE (%)'].iloc[0]:.2f}% WAPE. The upgraded app's restored default configuration (350 trees, leaf size 2, max features 0.9) has {fixed_all.loc[(fixed_all['Missing policy'] == 'Assume zero activity') & (fixed_all['Model'] == 'Extra Trees (restored original settings)'), 'WAPE (%)'].iloc[0]:.2f}% WAPE, compared with {original['WAPE (%)']:.2f}% for the former app implementation on the same six-month window. The modular app uses an expanded feature and imputation pipeline, so it is not identical to the old implementation even though the estimator settings are restored.",
        "- The Gradio default model is now Extra Trees with the original 350-tree settings; in this project's pipeline, the final holdout metric for that manually selected model is calculated with those same settings. Automatic rolling-origin selection remains optional.",
        "- Validation can favor different models at different horizons. Use a model chosen for the intended horizon, missing-month policy, and validation period.",
        "- Keep-missing and zero-fill are different business assumptions, so do not compare their metrics as if the targets were identical. Compare models within the same policy and dates.",
        "- The dataset is 84 calendar months long with seven source months absent. Even five rolling origins provide limited evidence; rerun validation when more history becomes available.",
        "- WAPE measures aggregate absolute error relative to aggregate actual volume; MAPE excludes zero actuals here. Positive bias means overforecasting and negative bias means underforecasting.",
        "- The supplied dataset includes vendor and purchasing details. Review whether it is permitted to publish before committing the CSV to a public GitHub repository.",
        "", "## Reproduction", "",
        "From the project directory, run `python scripts/benchmark.py`. This regenerates `MODEL_EVALUATION.md`, `original_baseline.csv`, `model_comparison_holdout.csv`, `model_comparison_rolling_origin.csv`, and `model_tuning_candidates.csv`.",
    ]
    (ROOT / "MODEL_EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print('Original baseline:', original)
    print('\nFinal holdout comparison:')
    print(fixed_all.to_string(index=False))
    print('\nBest rolling-origin config per family/horizon:')
    print(best_cv[["Missing policy", "Horizon (months)", "Model", "Parameters", "WAPE (%)", "Mean fold WAPE (%)", "Std fold WAPE (%)"]].to_string(index=False))
    print('\nCreated MODEL_EVALUATION.md and comparison CSV files.')

if __name__ == "__main__":
    main()
