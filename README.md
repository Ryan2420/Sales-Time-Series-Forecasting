# Achat Time Series Lab

A local-first purchasing analytics and monthly forecasting application with an interactive Gradio dashboard. It aggregates transaction-level data into calendar-month totals, checks data quality, compares ML models with naive baselines, performs chronological validation, and forecasts future months.

## Features

- Upload any compatible CSV and dynamically choose its date and numeric target columns.
- Preview the first 12 rows and inspect invalid records and absent calendar months.
- Explicitly choose whether months absent from the CSV remain unknown or are treated as zero activity.
- Compare **Naive**, **Seasonal Naive**, **Extra Trees**, **Hist Gradient Boosting**, and **Ridge Regression**.
- Default forecasting selection is restored to the original **Extra Trees** setup (`n_estimators=350`, `min_samples_leaf=2`, `max_features=0.9`); automatic rolling-origin selection remains available as an option.
- Run a small, reproducible hyperparameter search on development-period rolling-origin folds, then evaluate each selected configuration on the untouched final holdout.
- Evaluate a recent chronological rolling one-step holdout and a multi-step rolling-origin validation. Rolling-origin model selection is run only on the development period before the final holdout.
- Report WAPE, MAE, RMSE, MAPE over non-zero actuals, signed forecast bias, fold-level WAPE variation, and timing.
- Forecast 1–12 future months and view history, holdout predictions, and future estimates on one chart.
- Download forecast values, validation-tuned holdout comparisons, rolling-origin prediction details, and all hyperparameter-candidate scores as CSV files.
- Keep source files unmodified; no paid API or cloud dependency is required.

## Windows setup

Open PowerShell in this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860` while the terminal stays running. Stop the server with `Ctrl+C` when finished.

If PowerShell blocks environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.

## Run automated tests

```powershell
python -m pip install pytest
python -m pytest -q
```

## Project structure

```text
achat_timeseries_gradio/
├── app.py                  # Gradio UI and orchestration
├── data_pipeline.py        # CSV reading, numeric/date parsing, aggregation
├── features.py             # Past-only lag and rolling features
├── models.py               # Baselines, estimators, recursive forecasts
├── evaluation.py           # Metrics, holdout, tuning, rolling-origin evaluation
├── scripts/
│   └── benchmark.py        # Reproduce the model report
├── reference/
│   └── timeserieshanaoui_original.ipynb
├── tests/
│   └── test_project.py     # Unit and behavior tests
├── cleaned_achat_data.csv  # Included sample dataset
├── requirements.txt
├── MODEL_EVALUATION.md     # Reproducible results for the included sample
├── model_comparison_holdout.csv
├── model_comparison_rolling_origin.csv
├── model_tuning_candidates.csv
└── .gitignore
```

## Methodology and interpretation

- Transactions are aggregated using sums by calendar month. Repeated transaction lines are not automatically deleted because they may be legitimate.
- Date/target records that cannot be parsed are excluded and counted in the quality summary.
- Missing calendar months remain `NaN` by default. The model's lag-feature imputer is fitted within each training period. A missing target is not silently scored as zero.
- Choose “Assume zero activity” only when a month missing from the CSV truly means zero purchasing activity.
- Features use only prior months. The rolling-origin evaluation fits on each historical prefix, then recursively forecasts the requested horizon without seeing actual targets in that horizon.
- The recent fixed-model holdout is a rolling one-step diagnostic: actual values are added to history as each test month becomes known, while the fitted model stays fixed. This differs from predicting several months at once.
- WAPE is total absolute error divided by total absolute actual volume. MAPE excludes zero actual values; for sparse/zero-heavy series, interpret MAPE cautiously. Bias is prediction minus actual, so positive bias indicates overforecasting.
- Small monthly datasets can make model rankings unstable. The final holdout is excluded from automatic model and parameter selection. Compare models on the same dates and missing-month policy, and do not treat one backtest score as a guarantee.
- The original app's 13.12% WAPE was obtained under its original features and zero-filled missing months. The current modular pipeline uses a somewhat different feature/imputation implementation, so selecting the original Extra Trees hyperparameters does not guarantee that exact score will recur. To compare fairly, use the same dates, target, and missing-month policy.
- This project forecasts a selected monthly aggregate. It does not currently forecast individual products, suppliers, or stockout risk unless the underlying data and features are extended for that task.

## Included data

The downloadable ZIP includes the cleaned sample CSV and starts with it automatically when present. The CSV is ignored by Git by default to help avoid accidental publication of purchasing data; only force-add it if you have permission to disclose it. If the file is absent, upload an approved CSV in the app.

The bundled cleaned dataset has 5,593 rows and 22 columns. `Date transfert` covers January 2019 through December 2025, with seven absent calendar months in the source. All columns and target values should be checked for business meaning before operational decisions.

## Privacy

The app runs locally by default and does not send data to an external API. Confirm that any real business or personal data is permitted to be used on the computer where the app runs before sharing the repository publicly. Consider excluding the dataset from GitHub if it contains sensitive or commercially confidential records.
