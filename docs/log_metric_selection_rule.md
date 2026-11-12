# Selection rule for the log-SFR metric (fixed before any new result)

Written 2026-11-12, 16:20 KST, and committed before any new candidate was scored under a log metric.

## Why this rule exists

On 2026-11-12 the organisers' FAQ (Q6) stated that the primary quantitative criterion is the RMSE in log-SFR space, used first to shortlist teams on an unlabeled evaluation dataset. Our Part II model, ENS_top3, was selected by linear RMSE, the metric of the template's `FINAL_SCORE`. This document fixes in advance how the final model is re-selected, so that the choice cannot drift with the results (review point 2 of 2026-11-11).

## What had already been seen

Before this rule was written, scores under the three definitions below were computed for five models only, on CV and on the validation split:

| model | D1 CV | D2 CV | D3 CV |
|---|---:|---:|---:|
| R0 (M01_lgbm_tw13_F5) | 0.1694 | 0.5653 | 0.1064 |
| AG3_F5 | 0.1815 | 0.7477 | 0.1208 |
| AG23_F5C_tweedie | 0.1781 | 0.7663 | 0.1223 |
| AG26_F5C_wmassive | 0.1915 | 0.8497 | 0.1347 |
| ENS_top3 | 0.1761 | 0.8087 | 0.1231 |

(These are the notebook retrains of 2026-11-10.) No other candidate has been scored under a log metric.

## Metric

For a transform g, the fold score is the RMSE of g(prediction) − g(true SFR) over the galaxies of the fold; the CV score is the mean over the five folds `FOLDS` of the notebook (11 strata, seed 42). Predictions are clipped at 0 first.

- **D1**: g = log10, galaxies with SFR > 0 only, predictions floored at 1e-3.
- **D2**: g(y) = log10(y + 1e-3), all galaxies.
- **D3**: g(y) = log10(1 + y), all galaxies.

The **primary metric** is the organisers' definition if they give it before the decision deadline, 2026-11-14 12:00 KST; if it differs from D1–D3, it replaces them. If there is no answer by then, the primary metric is **D2**: FAQ Q4 says SFR = 0 is a valid target that is not removed, so the metric must give zeros a finite value, and an offset below the smallest positive SFR (0.0073) keeps the order of all values. D1, D2 and D3 are always reported.

## Candidates

All candidates use the template split and the same five folds; their scores come from out-of-fold (OOF) predictions.

- **A. The 139 project candidates** whose OOF predictions are stored in `results/oof/`.
- **B1. Direct log regression:** LightGBM with squared error on g(y), F5 and F5C, with the untuned R0 settings (learning rate 0.05, ≤ 1000 trees, early stopping on a stratified 10 % hold-out, 31 leaves).
- **B2. Hurdle:** a LightGBM classifier for P(SFR > 0) and a LightGBM regression on g(y) of the galaxies with SFR > 0, same settings, F5 and F5C. With p the classifier probability and m the regression output, the prediction is g⁻¹((1 − p)·g(0) + p·m) for D2 and D3, and g⁻¹(m) for D1 (zeros are not scored in D1).
- **B3. Log-space recalibration** of a stored candidate: isotonic regression of g(y) on g(prediction), fitted on the training folds of each outer fold.
- **C. AutoGluon on g(y)** (`best_quality`, as in the notebook), F5 and/or F5C, at most two runs, and only if a B candidate beats every A candidate under the primary metric.
- **D. Equal-weight averages in g-space** of the top k ≤ 3 candidates, with the members chosen inside each outer fold.

No hyperparameter is tuned beyond the settings listed here.

## Selection

1. Rank all candidates by CV under the primary metric.
2. **One-standard-error rule.** Among the candidates whose paired difference to the best (mean of the five fold differences) is at most its standard error (sd / √5), choose the simplest. Complexity order: single LightGBM (R0 type, B1) < hurdle (B2) < recalibrated candidate (B3) < single AutoGluon run < ensemble (D or an A ensemble). Equal complexity: lower CV wins.
3. **Nested estimate.** Steps 1–2 are repeated with one fold held out and the chosen candidate scored on that fold; the five scores and the five choices are reported.
4. The Tibshirani & Tibshirani (2009) bias estimate of the minimum CV score is reported.
5. The 20 % validation split is scored **once**, after the choice, for the chosen model and the references (R0, ENS_top3, the template HGB baseline). It is never used to choose or tune anything.
6. Every change to this document after the first new score is computed is recorded below, with date and reason, and disclosed in the notebook.

## Change log

(none)
