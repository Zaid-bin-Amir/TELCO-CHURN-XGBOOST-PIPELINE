# Tuned, Pipelined Churn Classification with Imbalance Handling and Error Analysis

Week 7 task. Telco Customer Churn, binary classification, about 26.5% churners.

## Project layout

```
churn_project/
  Churn_Project_Colab.ipynb   run this in Google Colab
  final_churn_model.py        same project as one script
  requirements.txt
  data/
    WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Run in Google Colab

1. Open https://colab.research.google.com and choose File > Upload notebook.
2. Upload `Churn_Project_Colab.ipynb`.
3. Choose Runtime > Run all.
4. When the upload cell asks for a file, select `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the `data` folder.
5. The last cell downloads `churn_results.zip` containing everything the run produced.

Total runtime is roughly 5 to 15 minutes depending on the Colab machine.

## Run locally

```
pip install -r requirements.txt
python final_churn_model.py
```

## What the run produces

| File | Content |
| --- | --- |
| `THRESHOLD_DECISION.md` | Chosen operating threshold, business rule, strategy comparison, hold-out numbers |
| `ERROR_ANALYSIS.md` | Ten misclassified customers, grouping, patterns of what the model gets wrong |
| `outputs/pr_curves.png` | Precision-recall curves with default and chosen thresholds marked |
| `outputs/threshold_tradeoff.png` | Precision, recall and F1 against the threshold |
| `outputs/model_comparison.csv` | Metrics for every imbalance strategy |
| `outputs/segment_error_rates.csv` | False negative and false positive rates per segment |
| `outputs/misclassified_sample.csv` | The ten sampled misclassified customers |
| `outputs/holdout_predictions.csv` | Probability and prediction for every hold-out customer |
| `models/churn_pipeline.joblib` | Fitted pipeline, threshold and strategy name |

## Method

- **Single Pipeline:** `ColumnTransformer` (median imputation and scaling for numeric columns, most-frequent imputation and one-hot encoding for categorical columns) followed by `XGBClassifier`, all inside one `Pipeline`.
- **Data fixes:** `TotalCharges` is converted to numeric, with blanks becoming missing values that the in-pipeline imputer fills. `customerID` is dropped from the features.
- **Split:** 80/20 stratified split. The 20% hold-out set is not used for tuning or for choosing the threshold.
- **Tuning:** `RandomizedSearchCV`, 5-fold stratified, 25 candidates, scored by PR-AUC, over six hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`.
- **Imbalance strategies compared:** an unweighted baseline, `scale_pos_weight`, and random oversampling of the positive class. Oversampling lives inside the model's `fit`, so it only ever touches training folds and never the validation fold or hold-out set.
- **Strategy selection:** highest cross-validated PR-AUC on training data among the two imbalance strategies.
- **Threshold:** chosen from out-of-fold training predictions as the highest-precision point that keeps recall at or above 85%, then applied once to the hold-out set.
- **Error analysis:** ten misclassified hold-out customers are sampled, grouped by shared traits, and false negative and false positive rates are computed per segment to find patterns.

All settings (seed, number of folds, number of search candidates, recall target) are at the top of the script in the configuration block.
