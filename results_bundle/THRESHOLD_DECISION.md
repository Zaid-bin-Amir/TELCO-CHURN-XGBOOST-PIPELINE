# Threshold Decision

## Business preference

The model is used to decide which customers receive a retention offer. Missing a customer who is about to leave (a false negative) costs that customer's future revenue, while contacting a customer who would have stayed (a false positive) only costs one offer. Recall is therefore prioritised over precision.

Decision rule: choose the threshold with the highest precision among all thresholds that keep recall at or above 85%.

## Imbalance strategy comparison

All three strategies were tuned with the same 5-fold stratified RandomizedSearchCV (25 candidates, scored by PR-AUC). Resampling happens only inside the training folds, so the hold-out set is never resampled. Metrics below are on the untouched hold-out set at the default 0.5 threshold. The strategy was selected by cross-validated PR-AUC on training data only.

| strategy | cv_pr_auc | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.6724 | 0.6760 | 0.5187 | 0.5870 | 0.8487 | 0.6660 |
| scale_pos_weight | 0.6713 | 0.5190 | 0.8021 | 0.6303 | 0.8476 | 0.6645 |
| random_oversampling | 0.6698 | 0.5274 | 0.7968 | 0.6347 | 0.8485 | 0.6646 |

Selected strategy: **scale_pos_weight**

Best hyperparameters:

- `colsample_bytree`: 0.85404
- `learning_rate`: 0.01064
- `max_depth`: 3
- `min_child_weight`: 8
- `n_estimators`: 563
- `subsample`: 0.68494

## How the threshold was chosen

The threshold was picked from out-of-fold predicted probabilities on the training data, so the hold-out set played no part in the choice. It was then applied once to the hold-out set.

Chosen threshold: **0.4328** (lower than the default 0.50).

| setting | threshold | precision | recall | f1 | tp | fp | fn | tn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Train out-of-fold, default | 0.5000 | 0.5164 | 0.8007 | 0.6279 | 1197 | 1121 | 298 | 3018 |
| Train out-of-fold, chosen | 0.4328 | 0.4905 | 0.8502 | 0.6221 | 1271 | 1320 | 224 | 2819 |

## Hold-out results

| setting | threshold | precision | recall | f1 | tp | fp | fn | tn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hold-out, default | 0.5000 | 0.5190 | 0.8021 | 0.6303 | 300 | 278 | 74 | 757 |
| Hold-out, chosen | 0.4328 | 0.4823 | 0.8396 | 0.6127 | 314 | 337 | 60 | 698 |

Hold-out size: 1409 customers, 374 of them churners.

## What the threshold implies

Compared with the default threshold, 0.4328 flags +73 customers on the hold-out set and changes the number of caught churners by +14. The model's probabilities are not calibrated, particularly when class weighting is used, so the threshold is a ranking cutoff and should not be read as a literal probability of churn. If a retention offer is cheap relative to lost revenue the threshold should move down, and if offers are expensive it should move up.

## Figures

- `outputs/pr_curves.png`: precision-recall curves with the default and chosen operating points marked.
- `outputs/threshold_tradeoff.png`: precision, recall and F1 against the threshold on out-of-fold predictions.

## Caveats

Recall on the hold-out set can differ from the 85% target because the threshold was fitted on different data. With 374 churners in the hold-out set, small differences in recall are within normal sampling noise.
