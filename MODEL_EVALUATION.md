# Model evaluation report — bundled purchasing data

## Dataset audit

- Raw rows: 5,593
- Valid date + target rows: 5,593
- Exact duplicate transaction rows (not removed): 0
- Target: `Quantité`; date: `Date transfert`
- Calendar span: 2019-01 to 2025-12 (84 months)
- Source months absent: 7 (2019-11, 2020-06, 2021-09, 2022-01, 2022-02, 2022-03, 2024-01)
- Transaction target IQR outlier flags: 458; this is a review heuristic, not an automatic deletion rule.

## Original app baseline

The former app's Extra Trees configuration (350 trees, original feature set) was re-run on the final six months with missing months explicitly filled with zero. It reproduces the prior reported baseline. This holdout is the same final six-month window used below.

| Metric | Original app baseline |
|---|---:|
| MAE | 2,930,640.59 |
| RMSE | 3,116,399.73 |
| WAPE | 13.1248% |
| MAPE | 13.8139% |

## Hyperparameter tuning

A compact, deterministic grid is evaluated on identical rolling-origin folds in the development period, which ends before the final six-month test window. The search is intentionally small for this 84-month dataset. The best configuration per model family is then evaluated on the final holdout; the holdout is not used to choose parameters. The Gradio forecast control now defaults to the original Extra Trees settings (`n_estimators=350`, `min_samples_leaf=2`, `max_features=0.9`), while automatic rolling-origin selection remains optional.

| Model family | Candidate configurations |
|---|---|
| Naive | Fixed baseline |
| Seasonal Naive | Fixed baseline, with fallback if the same month last year is missing |
| Extra Trees | 150/250/350 trees; `min_samples_leaf` 2/4; includes original 350-tree baseline |
| Hist Gradient Boosting | 120/160 iterations; `max_leaf_nodes` 5/7 |
| Ridge Regression | alpha 1 / 10 / 50 |

## Final six-month chronological holdout

All model families are evaluated on the same final six calendar months (July–December 2025). For each family, its hyperparameters were selected on development-period rolling-origin folds at a three-month horizon. These holdout scores are a separate final diagnostic, not a tuning target.

### Missing policy: Assume zero activity

| Model | Selected parameters | WAPE | MAE | RMSE | MAPE | Bias |
|---|---|---:|---:|---:|---:|---:|
| Extra Trees (restored original settings) | `{"max_features":0.9,"min_samples_leaf":2,"n_estimators":350}` | 11.71% | 2,613,890.13 | 2,744,645.76 | 11.56% | -1,327,768.94 |
| Extra Trees | `{"max_features":0.9,"min_samples_leaf":4,"n_estimators":250}` | 13.93% | 3,111,551.49 | 3,318,711.06 | 13.78% | -1,449,068.77 |
| Ridge Regression | `{"alpha":1.0}` | 17.44% | 3,894,437.52 | 5,075,653.45 | 19.05% | 1,093,203.00 |
| Hist Gradient Boosting | `{"l2_regularization":1.0,"learning_rate":0.05,"max_iter":120,"max_leaf_nodes":5}` | 17.66% | 3,944,401.99 | 4,258,839.88 | 17.38% | -1,926,866.14 |
| Naive | `{}` | 20.12% | 4,492,492.00 | 5,318,230.44 | 21.73% | 2,454,875.33 |
| Seasonal Naive | `{}` | 22.41% | 5,004,949.67 | 5,757,685.67 | 21.06% | -3,954,813.00 |

### Missing policy: Keep missing

| Model | Selected parameters | WAPE | MAE | RMSE | MAPE | Bias |
|---|---|---:|---:|---:|---:|---:|
| Extra Trees (restored original settings) | `{"max_features":0.9,"min_samples_leaf":2,"n_estimators":350}` | 12.79% | 2,856,838.62 | 3,572,017.51 | 14.53% | 1,035,704.72 |
| Extra Trees | `{"max_features":0.9,"min_samples_leaf":4,"n_estimators":250}` | 14.19% | 3,168,612.95 | 3,423,166.54 | 15.09% | -65,738.65 |
| Ridge Regression | `{"alpha":10.0}` | 16.80% | 3,751,246.06 | 5,294,207.13 | 19.68% | 2,253,427.11 |
| Hist Gradient Boosting | `{"l2_regularization":1.0,"learning_rate":0.05,"max_iter":120,"max_leaf_nodes":5}` | 17.28% | 3,858,212.50 | 4,001,671.34 | 17.57% | -1,334,910.06 |
| Naive | `{}` | 20.12% | 4,492,492.00 | 5,318,230.44 | 21.73% | 2,454,875.33 |
| Seasonal Naive | `{}` | 22.41% | 5,004,949.67 | 5,757,685.67 | 21.06% | -3,954,813.00 |

## Rolling-origin validation results

Each fold trains on an historical prefix and recursively forecasts 1, 3, or 6 months without receiving actual values within the forecast window. The final six months are excluded from every validation fold used for model selection. Metrics aggregate all scoreable prediction points; the standard deviation of fold WAPE indicates instability across origins.

| Missing policy | Horizon | Model | Selected parameters | WAPE | MAE | RMSE | MAPE | Mean fold WAPE | Std fold WAPE | Origins |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| Assume zero activity | 1 | Naive | `{}` | 32.78% | 8,178,305.20 | 12,875,376.98 | 31.34% | 31.34% | 39.23% | 5 |
| Assume zero activity | 1 | Ridge Regression | `{"alpha":10.0}` | 38.14% | 9,513,988.80 | 12,506,000.57 | 33.97% | 33.97% | 30.96% | 5 |
| Assume zero activity | 1 | Hist Gradient Boosting | `{"l2_regularization":1.0,"learning_rate":0.05,"max_iter":120,"max_leaf_nodes":5}` | 43.03% | 10,735,015.68 | 13,264,144.27 | 38.36% | 38.36% | 26.87% | 5 |
| Assume zero activity | 1 | Extra Trees | `{"max_features":0.9,"min_samples_leaf":4,"n_estimators":250}` | 43.17% | 10,770,522.08 | 13,843,498.51 | 36.56% | 36.56% | 32.96% | 5 |
| Assume zero activity | 1 | Seasonal Naive | `{}` | 44.83% | 11,183,295.20 | 12,765,170.90 | 50.09% | 50.09% | 33.61% | 5 |
| Assume zero activity | 3 | Ridge Regression | `{"alpha":1.0}` | 35.79% | 7,965,247.34 | 9,841,571.58 | 33.19% | 36.06% | 14.54% | 5 |
| Assume zero activity | 3 | Extra Trees | `{"max_features":0.9,"min_samples_leaf":4,"n_estimators":250}` | 39.55% | 8,802,113.16 | 11,011,278.09 | 36.95% | 39.07% | 22.95% | 5 |
| Assume zero activity | 3 | Hist Gradient Boosting | `{"l2_regularization":1.0,"learning_rate":0.05,"max_iter":120,"max_leaf_nodes":5}` | 39.64% | 8,821,025.69 | 11,460,822.37 | 36.49% | 38.64% | 25.12% | 5 |
| Assume zero activity | 3 | Naive | `{}` | 44.48% | 9,900,124.90 | 12,718,472.12 | 42.90% | 43.91% | 35.64% | 5 |
| Assume zero activity | 3 | Seasonal Naive | `{}` | 46.27% | 10,296,693.97 | 12,298,869.00 | 47.39% | 47.27% | 32.40% | 5 |
| Assume zero activity | 6 | Extra Trees | `{"max_features":0.9,"min_samples_leaf":2,"n_estimators":150}` | 42.78% | 8,279,458.61 | 10,793,265.67 | 52.81% | 43.59% | 18.91% | 5 |
| Assume zero activity | 6 | Hist Gradient Boosting | `{"l2_regularization":1.0,"learning_rate":0.05,"max_iter":120,"max_leaf_nodes":5}` | 43.74% | 8,466,340.17 | 11,167,803.72 | 49.05% | 44.20% | 18.90% | 5 |
| Assume zero activity | 6 | Ridge Regression | `{"alpha":1.0}` | 43.78% | 8,473,475.47 | 10,596,615.17 | 62.23% | 44.19% | 8.23% | 5 |
| Assume zero activity | 6 | Seasonal Naive | `{}` | 48.73% | 9,431,389.72 | 11,181,007.31 | 51.29% | 48.50% | 16.73% | 5 |
| Assume zero activity | 6 | Naive | `{}` | 59.85% | 11,584,503.48 | 13,572,670.40 | 77.89% | 62.28% | 25.74% | 5 |
| Keep missing | 1 | Naive | `{}` | 28.72% | 7,163,793.20 | 10,732,345.60 | 27.65% | 27.65% | 31.19% | 5 |
| Keep missing | 1 | Ridge Regression | `{"alpha":10.0}` | 35.29% | 8,802,923.56 | 11,643,258.00 | 31.37% | 31.37% | 27.74% | 5 |
| Keep missing | 1 | Extra Trees | `{"max_features":0.9,"min_samples_leaf":2,"n_estimators":350}` | 38.23% | 9,536,668.70 | 11,989,742.69 | 32.92% | 32.92% | 24.92% | 5 |
| Keep missing | 1 | Seasonal Naive | `{}` | 38.47% | 9,596,219.20 | 11,864,448.64 | 35.60% | 35.60% | 19.27% | 5 |
| Keep missing | 1 | Hist Gradient Boosting | `{"l2_regularization":1.0,"learning_rate":0.05,"max_iter":120,"max_leaf_nodes":5}` | 41.04% | 10,237,845.85 | 12,615,899.04 | 35.56% | 35.56% | 26.00% | 5 |
| Keep missing | 3 | Ridge Regression | `{"alpha":10.0}` | 30.39% | 6,762,478.64 | 8,826,985.06 | 27.61% | 29.83% | 13.93% | 5 |
| Keep missing | 3 | Extra Trees | `{"max_features":0.9,"min_samples_leaf":4,"n_estimators":250}` | 34.96% | 7,780,402.59 | 9,756,272.57 | 31.92% | 33.62% | 15.40% | 5 |
| Keep missing | 3 | Hist Gradient Boosting | `{"l2_regularization":1.0,"learning_rate":0.05,"max_iter":120,"max_leaf_nodes":5}` | 37.08% | 8,253,113.96 | 10,623,436.94 | 33.78% | 36.05% | 19.72% | 5 |
| Keep missing | 3 | Seasonal Naive | `{}` | 39.70% | 8,835,919.97 | 10,783,114.58 | 37.45% | 38.19% | 16.27% | 5 |
| Keep missing | 3 | Naive | `{}` | 39.93% | 8,885,612.90 | 11,374,610.98 | 36.70% | 38.42% | 25.52% | 5 |
| Keep missing | 6 | Extra Trees | `{"max_features":0.9,"min_samples_leaf":4,"n_estimators":250}` | 38.22% | 7,652,189.33 | 9,821,526.98 | 57.01% | 38.25% | 6.38% | 5 |
| Keep missing | 6 | Hist Gradient Boosting | `{"l2_regularization":1.0,"learning_rate":0.05,"max_iter":120,"max_leaf_nodes":5}` | 41.31% | 8,271,881.06 | 10,997,241.96 | 56.13% | 41.48% | 13.95% | 5 |
| Keep missing | 6 | Seasonal Naive | `{}` | 43.24% | 8,658,081.78 | 10,406,992.99 | 52.02% | 42.51% | 9.96% | 5 |
| Keep missing | 6 | Ridge Regression | `{"alpha":50.0}` | 43.64% | 8,737,897.32 | 10,681,121.68 | 61.18% | 44.06% | 7.26% | 5 |
| Keep missing | 6 | Naive | `{}` | 51.95% | 10,401,589.81 | 12,472,154.98 | 69.34% | 53.16% | 17.17% | 5 |

## Interpretation

- The original app's WAPE is a baseline for one recent six-month rolling one-step test, not a universal forecast accuracy percentage.
- On the zero-fill final holdout, the current pipeline's validation-selected Extra Trees configuration has 13.93% WAPE. The upgraded app's restored default configuration (350 trees, leaf size 2, max features 0.9) has 11.71% WAPE, compared with 13.12% for the former app implementation on the same six-month window. The modular app uses an expanded feature and imputation pipeline, so it is not identical to the old implementation even though the estimator settings are restored.
- The Gradio default model is now Extra Trees with the original 350-tree settings; in this project's pipeline, the final holdout metric for that manually selected model is calculated with those same settings. Automatic rolling-origin selection remains optional.
- Validation can favor different models at different horizons. Use a model chosen for the intended horizon, missing-month policy, and validation period.
- Keep-missing and zero-fill are different business assumptions, so do not compare their metrics as if the targets were identical. Compare models within the same policy and dates.
- The dataset is 84 calendar months long with seven source months absent. Even five rolling origins provide limited evidence; rerun validation when more history becomes available.
- WAPE measures aggregate absolute error relative to aggregate actual volume; MAPE excludes zero actuals here. Positive bias means overforecasting and negative bias means underforecasting.
- The supplied dataset includes vendor and purchasing details. Review whether it is permitted to publish before committing the CSV to a public GitHub repository.

## Reproduction

From the project directory, run `python scripts/benchmark.py`. This regenerates `MODEL_EVALUATION.md`, `original_baseline.csv`, `model_comparison_holdout.csv`, `model_comparison_rolling_origin.csv`, and `model_tuning_candidates.csv`.
