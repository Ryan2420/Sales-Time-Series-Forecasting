"""Achat Time Series Lab: local-first forecasting dashboard for monthly purchasing data."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_pipeline import (
    DEFAULT_DATE_CANDIDATES, build_monthly_series, date_columns,
    infer_column, numeric_columns, read_csv,
)
from evaluation import compare_fixed_holdout, fixed_holdout_backtest, rolling_origin_evaluation
from models import AUTO_MODEL_SELECTION, FORECAST_MODEL_CHOICES, MODEL_NAMES, default_parameters, forecast

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "cleaned_achat_data.csv"
APP_CSS = """
.hero {padding: 24px 26px; border-radius: 18px; background: linear-gradient(115deg,#10253f,#155e75 70%,#5b3c88); color:#fff; margin-bottom:12px}
.hero h1 {font-size:30px; line-height:1.15; margin:0 0 8px 0;}
.hero p {font-size:14px; opacity:.92; margin:0;}
.section-note {font-size:13px; opacity:.8}
footer {display:none !important}
"""


def inspect_upload(file_path):
    """Update column selectors and show a small data preview when a CSV is selected."""
    if not file_path:
        return (
            gr.update(choices=["(Auto-detect)"], value="(Auto-detect)"),
            gr.update(choices=[], value=None),
            pd.DataFrame(),
            "Upload a CSV file to inspect its columns.",
        )
    try:
        frame = read_csv(file_path)
        dates = date_columns(frame)
        numeric = numeric_columns(frame)
        date_choices = ["(Auto-detect)"] + dates
        chosen_date = infer_column(frame, DEFAULT_DATE_CANDIDATES, "date")
        target_choices = numeric
        chosen_target = "Quantité" if "Quantité" in numeric else infer_column(frame, [], "numeric")
        summary = (
            f"**Loaded:** {Path(file_path).name} · **Rows:** {len(frame):,} · "
            f"**Columns:** {len(frame.columns)} · **Date candidates:** {len(dates)} · "
            f"**Numeric candidates:** {len(numeric)}"
        )
        return (
            gr.update(choices=date_choices, value=chosen_date or "(Auto-detect)"),
            gr.update(choices=target_choices, value=chosen_target),
            frame.head(12),
            summary,
        )
    except Exception as exc:
        return (
            gr.update(choices=["(Auto-detect)"], value="(Auto-detect)"),
            gr.update(choices=[], value=None),
            pd.DataFrame(),
            f"**Could not inspect the CSV:** {type(exc).__name__}: {exc}",
        )


def format_num(value, digits=2):
    try:
        return f"{float(value):,.{digits}f}" if np.isfinite(float(value)) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def csv_download(frame: pd.DataFrame, prefix: str) -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", prefix=f"{prefix}_", suffix=".csv", delete=False, encoding="utf-8-sig", newline="")
    handle.close()
    frame.to_csv(handle.name, index=False, encoding="utf-8-sig")
    return handle.name


def plot_dashboard(series, backtest_df, forecast_series, title):
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.plot(series.index, series.values, label="Historical actuals", linewidth=2.2, color="#2563eb")
    if backtest_df is not None and not backtest_df.empty:
        dates = pd.to_datetime(backtest_df["Date"])
        actual = pd.to_numeric(backtest_df["Actual"], errors="coerce")
        predicted = pd.to_numeric(backtest_df["Predicted"], errors="coerce")
        ax.scatter(dates, actual, label="Holdout actual", marker="o", s=34, color="#15803d", zorder=4)
        ax.plot(dates, predicted, label="Holdout predictions", linestyle="--", marker="x", linewidth=1.8, color="#ea580c")
    if forecast_series is not None and len(forecast_series):
        ax.plot(forecast_series.index, forecast_series.values, label="Future forecast", marker="o", linestyle="--", linewidth=2.2, color="#7c3aed")
        ax.axvline(series.index[-1], linestyle=":", linewidth=1.2, color="#64748b", label="Forecast starts")
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly total")
    ax.grid(True, alpha=0.24)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(loc="best", ncols=2, frameon=False)
    fig.tight_layout()
    return fig


def analyze(file_path, date_choice, target_col, missing_policy, dayfirst, model_name, horizon, test_months, cv_origins):
    if not file_path:
        raise gr.Error("Select a CSV file first.")
    if not target_col:
        raise gr.Error("Choose a numeric target column.")
    try:
        raw = read_csv(file_path)
        date_col = date_choice
        if not date_col or date_col == "(Auto-detect)":
            date_col = infer_column(raw, DEFAULT_DATE_CANDIDATES, "date")
        if date_col is None:
            raise ValueError("No date-like column was detected. Select the correct date column manually.")
        series, meta = build_monthly_series(raw, date_col, target_col, missing_policy=missing_policy, dayfirst=bool(dayfirst))
        if len(series) < 36 or int(series.notna().sum()) < 24:
            raise ValueError(
                f"This series has {len(series)} calendar months and {int(series.notna().sum())} observed totals. "
                "At least 36 calendar months and 24 observed monthly totals are recommended."
            )

        horizon = int(horizon)
        test_months = int(test_months)
        cv_origins = int(cv_origins)
        started = time.perf_counter()

        # Reserve the latest chronological holdout entirely outside model and hyperparameter selection.
        development_series = series.iloc[:-test_months]
        tuning_summary, rolling_details = rolling_origin_evaluation(
            development_series, model_names=MODEL_NAMES, horizon=horizon, max_origins=cv_origins,
        )
        valid_cv = tuning_summary.loc[np.isfinite(pd.to_numeric(tuning_summary["WAPE (%)"], errors="coerce"))]
        if valid_cv.empty and model_name == AUTO_MODEL_SELECTION:
            raise ValueError(
                "Rolling-origin validation could not calculate WAPE for any model. Choose a specific model or review the target values."
            )
        if valid_cv.empty:
            raise ValueError("Rolling-origin validation has no scoreable windows for this dataset and forecast horizon.")

        best_config_rows = (
            valid_cv.sort_values("WAPE (%)")
            .groupby("Model", sort=False, as_index=False)
            .first()
        )
        best_params_by_model = {
            row["Model"]: json.loads(row["Parameters"])
            for _, row in best_config_rows.iterrows()
        }
        rolling_summary = best_config_rows.sort_values("WAPE (%)", na_position="last").reset_index(drop=True)
        cv_winner_row = valid_cv.sort_values("WAPE (%)").iloc[0]
        cv_winner = str(cv_winner_row["Model"])
        chosen_model = cv_winner if model_name == AUTO_MODEL_SELECTION else model_name

        if model_name == AUTO_MODEL_SELECTION:
            # Automatic mode uses the winning model/configuration selected on development folds.
            chosen_params = best_params_by_model.get(chosen_model, {})
        elif chosen_model == "Extra Trees":
            # Manual Extra Trees selection restores the original app's baseline settings.
            chosen_params = default_parameters("Extra Trees")
        else:
            # For other manually selected families, use that family's best development-CV candidate.
            chosen_params = best_params_by_model.get(chosen_model, {})

        # Compare families on the same final holdout. The selected manual Extra Trees baseline
        # is evaluated with its actual 350-tree settings, so displayed metrics match the forecast model.
        holdout_params = dict(best_params_by_model)
        holdout_params[chosen_model] = chosen_params
        fixed_comparison, fixed_details = compare_fixed_holdout(
            series, test_months=test_months, model_names=MODEL_NAMES,
            model_params_by_name=holdout_params,
        )
        selected_row = fixed_comparison.loc[fixed_comparison["Model"] == chosen_model].iloc[0].to_dict()
        selected_metrics = selected_row
        selected_holdout = fixed_details[chosen_model].copy()
        # After selection and evaluation, refit the selected configuration on all available history for future forecasts.
        forecast_series, _ = forecast(series, horizon, chosen_model, nonnegative=None, model_params=chosen_params)
        elapsed = time.perf_counter() - started

        missing_text = ", ".join(meta["missing_months"]) if meta["missing_months"] else "None"
        best_cv = cv_winner_row if not valid_cv.empty else None
        summary = f"""## Run summary

**Target:** `{target_col}` · **Model used for forecast:** **{chosen_model}** · **Forecast horizon:** {horizon} month(s)

**Model selection:** {f"automatic rolling-origin winner — {cv_winner}" if model_name == AUTO_MODEL_SELECTION else ("manual selection — original Extra Trees baseline configuration" if chosen_model == "Extra Trees" else f"manual model-family selection — {chosen_model} using its best development-CV configuration")}. The final {test_months}-month holdout is kept separate from automatic model and parameter selection. Manual Extra Trees mode restores the original 350-tree settings; its displayed holdout metrics are calculated with those exact settings.

**Selected hyperparameters:** `{json.dumps(chosen_params, sort_keys=True) if chosen_params else "baseline (no fitted parameters)"}`

| Data quality | Result |
|---|---:|
| Raw transaction rows | {meta['raw_rows']:,} |
| Valid date + target rows | {meta['valid_rows']:,} |
| Invalid date rows | {meta['invalid_date_rows']:,} |
| Invalid numeric target rows | {meta['invalid_target_rows']:,} |
| Calendar span | {meta['start']} → {meta['end']} ({meta['calendar_months']} months) |
| Months absent in source | {meta['missing_months_count']} |
| Missing-month policy | {missing_policy} |

**Absent months:** {missing_text}

### Recent chronological holdout
*{selected_metrics['Test months']} final calendar months; fixed model trained before the holdout, then evaluated one month ahead using prior actuals as they become available.*

| Metric | Value |
|---|---:|
| WAPE | {format_num(selected_metrics['WAPE (%)'])}% |
| MAE | {format_num(selected_metrics['MAE'])} |
| RMSE | {format_num(selected_metrics['RMSE'])} |
| MAPE (non-zero actuals only) | {format_num(selected_metrics['MAPE (%)'])}% |
| Mean bias (prediction − actual) | {format_num(selected_metrics['Bias'])} |
| Bias relative to actual volume | {format_num(selected_metrics['Bias (%)'])}% |
| Scored months | {selected_metrics['Points']} |

### Multi-step rolling-origin validation
*The model is refit at each origin and recursively forecasts {horizon} month(s). Results are aggregated across {rolling_summary['Origins'].max() if not rolling_summary.empty else 0} chronological origins drawn only from the development period before the final holdout; unknown actual months are excluded from metrics.*

"""
        if best_cv is not None:
            summary += (
                f"Lowest tuned rolling-origin WAPE in this run: **{best_cv['Model']} — "
                f"{format_num(best_cv['WAPE (%)'])}%** (parameters `{best_cv['Parameters']}`). This is a validation result, not a guarantee of future performance.\n\n"
            )
        summary += (
            f"**Runtime:** {elapsed:.1f} seconds. **Important:** WAPE is meaningful only when models are compared on the same dates and missing-month policy. "
            "Automatic selection is based only on earlier rolling-origin validation, not on the final holdout. For a manual model, select it explicitly and inspect the same validation table."
        )

        quality = f"""## Data-quality notes
- Dates and target values that could not be parsed were excluded from monthly aggregation.
- Exact duplicate rows detected: {meta['exact_duplicate_rows']:,}. Duplicates were **not** automatically removed; repeated lines/documents may be legitimate.
- Missing calendar months are **not treated as zero** unless you explicitly choose **Assume zero activity**.
- Under **Keep missing**, missing targets remain unknown, are excluded from model targets and score calculations, and missing lag features are imputed from each training fold only.
- An IQR-based check flags {meta['iqr_outlier_rows']:,} transaction target values outside [{format_num(meta['iqr_lower_fence'])}, {format_num(meta['iqr_upper_fence'])}]. This is only a review heuristic; flagged purchases are not automatically deleted.
- The observed transaction target range is {format_num(meta['target_min'])} to {format_num(meta['target_max'])}; median transaction target is {format_num(meta['target_median'])}.
"""

        forecast_table = forecast_series.rename_axis("Date").reset_index()
        forecast_table["Date"] = forecast_table["Date"].dt.strftime("%Y-%m")
        forecast_table["Forecast"] = forecast_table["Forecast"].round(2)
        forecast_table["Model"] = chosen_model
        forecast_table["Parameters"] = json.dumps(chosen_params, sort_keys=True)
        holdout_table = selected_holdout.copy()
        holdout_table["Date"] = pd.to_datetime(holdout_table["Date"]).dt.strftime("%Y-%m")
        for col in ["Actual", "Predicted", "Error", "Absolute error"]:
            holdout_table[col] = pd.to_numeric(holdout_table[col], errors="coerce").round(2)

        fixed_display = fixed_comparison.copy()
        rolling_display = rolling_summary.copy()
        tuning_display = tuning_summary.copy()
        for table in (fixed_display, rolling_display, tuning_display):
            for col in table.select_dtypes(include=[np.number]).columns:
                table[col] = table[col].round(3)

        cv_details_download = rolling_details.copy()
        cv_details_download["Origin date"] = pd.to_datetime(cv_details_download["Origin date"]).dt.strftime("%Y-%m")
        cv_details_download["Date"] = pd.to_datetime(cv_details_download["Date"]).dt.strftime("%Y-%m")

        fig = plot_dashboard(series, selected_holdout, forecast_series, f"{target_col}: history, evaluation and forecast")
        return (
            summary, quality, fig, forecast_table, holdout_table, fixed_display, rolling_display, tuning_display,
            csv_download(forecast_table, "achat_forecast"),
            csv_download(fixed_comparison, "achat_fixed_holdout_comparison"),
            csv_download(cv_details_download, "achat_rolling_origin_details"),
            csv_download(tuning_summary, "achat_hyperparameter_tuning"),
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Analysis failed: {type(exc).__name__}: {exc}") from exc


def build_app():
    default_path = str(DEFAULT_CSV) if DEFAULT_CSV.exists() else None
    date_defaults = ["(Auto-detect)", "Date transfert", "Date prévue"]
    target_defaults = [
        "Quantité", "montant total ht de la piece", "montant total  ttc",
        "Prix de revient HT total", "Montant total HT CDE", "Montant TTC CDE",
    ]
    with gr.Blocks(title="Achat Time Series Lab") as demo:
        gr.HTML("<div class='hero'><h1>📈 Achat Time Series Lab</h1><p>Purchasing intelligence · time-aware validation · practical monthly forecasts</p></div>")
        gr.Markdown(
            "Upload a transaction CSV or use the bundled sample. Choose a target, make missing-month assumptions explicit, "
            "then compare ML models against simple baselines before using a forecast operationally."
        )
        with gr.Row():
            file_input = gr.File(label="Purchasing data (CSV)", file_types=[".csv"], type="filepath", value=default_path, scale=2)
            with gr.Column(scale=3):
                upload_summary = gr.Markdown("Bundled dataset is ready." if default_path else "Upload a CSV to get started.")
                with gr.Row():
                    date_input = gr.Dropdown(label="Date column", choices=date_defaults, value="(Auto-detect)", allow_custom_value=True)
                    target_input = gr.Dropdown(label="Monthly target", choices=target_defaults, value="Quantité", allow_custom_value=True)
        with gr.Accordion("Preview & configuration", open=True):
            preview = gr.Dataframe(label="First 12 rows", interactive=False, max_height=260)
            with gr.Row():
                with gr.Column():
                    missing_policy = gr.Radio(
                        label="How should months absent from the CSV be treated?",
                        choices=["Keep missing", "Assume zero activity"], value="Keep missing",
                        info="Choose zero only if absence from the source truly means no purchasing activity.",
                    )
                    dayfirst_input = gr.Checkbox(
                        label="Interpret ambiguous dates as day-first (DD/MM/YYYY)", value=True,
                        info="ISO dates such as 2025-12-31 are parsed correctly either way.",
                    )
                model_input = gr.Dropdown(
                    label="Model for future forecast", choices=FORECAST_MODEL_CHOICES,
                    value="Extra Trees",
                    info="Defaults to the original baseline: 350 trees, min_samples_leaf=2, max_features=0.9. Choose Automatic to select by development-period rolling-origin WAPE.",
                )
            with gr.Row():
                horizon_input = gr.Slider(label="Future forecast horizon (months)", minimum=1, maximum=12, value=3, step=1)
                test_months_input = gr.Slider(label="Recent holdout window (months)", minimum=3, maximum=12, value=6, step=1)
                origins_input = gr.Slider(label="Rolling-origin windows", minimum=3, maximum=8, value=5, step=1)
        run_button = gr.Button("Run audit, compare models & forecast", variant="primary", size="lg")
        summary_output = gr.Markdown()
        quality_output = gr.Markdown()
        with gr.Tab("Forecast dashboard"):
            plot_output = gr.Plot(label="Monthly history, holdout results and forecast")
            forecast_output = gr.Dataframe(label="Future forecast", interactive=False)
            forecast_download = gr.File(label="Download forecast CSV")
        with gr.Tab("Recent holdout"):
            gr.Markdown("This rolling one-step holdout uses each newly observed month to inform the next prediction, while the fitted ML model remains fixed.")
            holdout_output = gr.Dataframe(label="Actual versus predicted", interactive=False)
            fixed_comparison_output = gr.Dataframe(label="Model comparison on the same recent holdout", interactive=False)
            holdout_download = gr.File(label="Download holdout comparison CSV")
        with gr.Tab("Rolling-origin validation"):
            gr.Markdown("Every fold trains on past data only and predicts the requested number of future months recursively. The final holdout is excluded from these model-selection folds.")
            rolling_output = gr.Dataframe(label="Best validation-tuned configuration per model family", interactive=False)
            rolling_download = gr.File(label="Download rolling-origin prediction details CSV")
        with gr.Tab("Hyperparameter tuning"):
            gr.Markdown("A compact candidate grid is compared on identical rolling-origin validation windows. The best candidate for each model family is then evaluated on the untouched recent holdout.")
            tuning_output = gr.Dataframe(label="All configurations tested", interactive=False)
            tuning_download = gr.File(label="Download hyperparameter tuning results CSV")
        run_button.click(
            fn=analyze,
            inputs=[file_input, date_input, target_input, missing_policy, dayfirst_input, model_input, horizon_input, test_months_input, origins_input],
            outputs=[summary_output, quality_output, plot_output, forecast_output, holdout_output,
                     fixed_comparison_output, rolling_output, tuning_output,
                     forecast_download, holdout_download, rolling_download, tuning_download],
        )
        file_input.change(
            fn=inspect_upload,
            inputs=[file_input],
            outputs=[date_input, target_input, preview, upload_summary],
        )
        if default_path:
            # Populate preview and selectors at app start; this does not run model training.
            demo.load(inspect_upload, inputs=[file_input], outputs=[date_input, target_input, preview, upload_summary])
        gr.Markdown(
            "---\n**Methodology:** transaction values are summed by calendar month; lag and rolling features use past data only; "
            "unknown targets are not scored as zero; model comparisons use identical forecast dates and the same missing-month policy. "
            "No forecast is a purchasing guarantee. Validate unusual or high-impact predictions with business context."
        )
    return demo


if __name__ == "__main__":
    build_app().launch(
        server_name="127.0.0.1", server_port=7862, show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="violet", neutral_hue="slate"),
        css=APP_CSS,
    )
