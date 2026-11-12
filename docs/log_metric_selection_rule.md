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

- **2026-11-12, after the re-rank of the 139 stored candidates (commit cd5e34c) and before any B candidate was computed. Clarifications, no change of substance.**
  1. *Complexity tier of the stored candidates.* The rule named only LightGBM for the first tier. Every stored candidate that is one fitted model (any family, including nested-tuned models and in-fold feature selection) belongs to the first tier, together with B1. Stored AutoGluon runs belong to the AutoGluon tier and stored ensembles to the ensemble tier.
  2. *B3 is applied to every stored candidate* (the rule said "a stored candidate"). The nested estimate counts these recalibrated versions as separate candidates.
  3. *B1 and B2 stay on F5 and F5C as written.* The re-rank showed in-fold selection from FULL (564 columns) at the top of D2, but adding FULL to B now would be a choice made after seeing results, so it is not done.
  4. *B1 and B2 under D1* are the same model (a regression on log10 SFR of the galaxies with SFR > 0, applied to all galaxies), so it is counted once.

- **2026-11-12, organisers' answer received (before any candidate was scored under it).** The final quantitative metric clips negative predictions to 0, applies a floor of 1e-13 to both the true and the predicted SFR, and computes the RMSE of log10[max(SFR, 1e-13)]. Following the rule ("if it differs from D1–D3, it replaces them"), this is now the primary metric, called **DS** below: g(y) = log10(max(y, 1e-13)), all galaxies. Zeros map to −13 and the smallest positive SFR to −2.14. B1 (regression on g(SFR)), B2 (hurdle with g(0) = −13 and the regression on g(SFR) = log10 SFR of the galaxies with SFR > 0), B3 and D are computed under DS exactly as defined above; nothing else changes. D1–D3 remain reported as secondary metrics.
- **2026-11-12, after the DS scores of A, B and D were computed.** (a) The condition for C holds under DS: the best B candidate (hurdle on F5C, CV 1.455) beats every stored candidate (best: 2.989). The two allowed AutoGluon runs are started: regression on g(SFR) = log10(max(SFR, 1e-13)) on F5C, then on F5 (`scripts/log_candidates_c.py`). They enter the selection in the AutoGluon tier. (b) *Deployment for the organisers' private test set*, fixed before the validation split is scored: the chosen pipeline is refit with the same settings on all 208,437 labelled galaxies (training and validation split), with early stopping again on a stratified 10 % hold-out. The validation score in the notebook still comes from the model trained on the training split only, as the template requires; the refit model is used only for the private test predictions.
- **2026-11-13, after both AutoGluon runs and before the validation split was scored.** (a) *Result of the rule under DS* (296 candidates, `scripts/log_select.py`): the lowest CV is D_top3_DS, 1.4356, whose members are C_AG_DS_F5C, C_AG_DS_F5 and B2_DS_F5C in every fold. No other candidate is within one SE of it (D_top2_DS +0.0011, SE 0.0008; C_AG_DS_F5C +0.0025, SE 0.0011), so the 1-SE choice is D_top3_DS. Nested estimate 1.4356 (D_top3_DS chosen with each fold held out), Tibshirani & Tibshirani bias 0.0003. (b) *Incident:* during C_AG_DS_F5 the laptop entered Modern Standby at 2026-11-12 20:21:30 KST (power source change), about two minutes before the run's time limit, and resumed at 23:54:49. The run then finished normally (16 models). Its wall-clock time, 18,270 s, includes 12,799 s of standby; 5,471 s is reported as its training time. (c) The deployment refit of entry 3b was started on 2026-11-13 at 00:06 (`scripts/log_refit_all.py`); the two log-target AutoGluon runs were moved into the notebook folder (`ag_runs/`), from which the notebook reloads them.

- **2026-11-13, deployment refit finished; validation split scored once.** (a) The refit of entry 3b finished at 03:28 KST (202 min; hurdle on F5C: 296 / 279 trees; C_AG_DS_F5C_all 11 models in 6,633 s, C_AG_DS_F5_all 17 models in 5,403 s; two Ray worker crashes logged, neither stopped a run). With 25 % more rows and the same time limit, the F5C refit fits fewer models than the training-split run (16). (b) The notebook was then executed once (04:01 KST). Its replay of the rule on its 12 candidates gives the same choice, D_top3 (CV 1.4356, nested 1.4356, bias 0.0003), and its B2_DS_F5C OOF predictions are identical to `results/log_metric/oof_B/B2_DS_F5C.parquet`. Validation log-RMSE (scored once, after the choice): D_top3_DS 1.4249; C_AG_DS_F5C 1.4242, C_AG_DS_F5 1.4308, B2_DS_F5C 1.4463, R0 3.3602, ENS_top3 3.7395. Nothing was changed after this score.
