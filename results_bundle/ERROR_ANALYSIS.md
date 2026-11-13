# Error Analysis

## Setup

- Model: `scale_pos_weight` strategy, decision threshold 0.4328
- Hold-out customers: 1409
- Confusion matrix: TP 314, FP 337, FN 60, TN 698
- Share of churners missed (false negative rate): 16.0%
- Share of stayers wrongly flagged (false positive rate): 32.6%

## Ten misclassified customers

The five most confident false negatives (churners given the lowest churn probability) and the five most confident false positives (stayers given the highest churn probability).

| customerID | error_type | proba | tenure | Contract | InternetService | PaymentMethod | MonthlyCharges | TechSupport |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8580-QVLOC | False Negative | 0.025 | 72 | Two year | DSL | Credit card (automatic) | 92.450 | Yes |
| 3400-ESFUW | False Negative | 0.035 | 55 | Two year | DSL | Bank transfer (automatic) | 57.550 | Yes |
| 6598-RFFVI | False Negative | 0.076 | 2 | One year | No | Credit card (automatic) | 19.300 | No internet service |
| 3705-PSNGL | False Negative | 0.083 | 45 | One year | No | Electronic check | 20.400 | No internet service |
| 4464-JCOLN | False Negative | 0.093 | 2 | One year | No | Mailed check | 19.850 | No internet service |
| 4927-WWOOZ | False Positive | 0.922 | 2 | Month-to-month | Fiber optic | Electronic check | 91.450 | No |
| 2545-EBUPK | False Positive | 0.921 | 2 | Month-to-month | Fiber optic | Electronic check | 84.050 | No |
| 1628-BIZYP | False Positive | 0.921 | 1 | Month-to-month | Fiber optic | Electronic check | 85.000 | No |
| 8161-QYMTT | False Positive | 0.920 | 7 | Month-to-month | Fiber optic | Electronic check | 94.100 | No |
| 6630-UJZMY | False Positive | 0.913 | 4 | Month-to-month | Fiber optic | Electronic check | 83.250 | No |

## Grouping by shared characteristics

- **False Negatives (5 customers):** median tenure 45 months, most common contract One year (3 of 5), internet service No (3 of 5), payment method Credit card (automatic) (2 of 5), median monthly charge 20.40, mean predicted churn probability 0.06.
- **False Positives (5 customers):** median tenure 2 months, most common contract Month-to-month (5 of 5), internet service Fiber optic (5 of 5), payment method Electronic check (5 of 5), median monthly charge 85.00, mean predicted churn probability 0.92.

Counts by error type, contract and internet service:

| error_type | Contract | InternetService | customers |
| --- | --- | --- | --- |
| False Negative | One year | No | 3 |
| False Negative | Two year | DSL | 2 |
| False Positive | Month-to-month | Fiber optic | 5 |

## Patterns across the full hold-out set

Segments below have at least 30 customers and at least 15 customers of the relevant class.

### Where the model misses churners

1. Churners with Contract = One year are missed 72% of the time (26 of 36), against 16% overall.
2. Churners with InternetService = No are missed 56% of the time (14 of 25), against 16% overall.
3. Churners with tenure_band = 49-72 months are missed 54% of the time (19 of 35), against 16% overall.

| feature | value | customers | churners | fn_rate | stayers | fp_rate |
| --- | --- | --- | --- | --- | --- | --- |
| Contract | One year | 300 | 36 | 0.722 | 264 | 0.125 |
| InternetService | No | 312 | 25 | 0.560 | 287 | 0.077 |
| tenure_band | 49-72 months | 450 | 35 | 0.543 | 415 | 0.116 |
| PaymentMethod | Bank transfer (automatic) | 300 | 57 | 0.316 | 243 | 0.173 |
| tenure_band | 25-48 months | 304 | 64 | 0.281 | 240 | 0.275 |
| PaperlessBilling | No | 569 | 100 | 0.260 | 469 | 0.188 |
| InternetService | DSL | 484 | 97 | 0.258 | 387 | 0.245 |
| PaymentMethod | Credit card (automatic) | 309 | 51 | 0.255 | 258 | 0.205 |

### Where the model raises false alarms

1. Customers who stay with Contract = Month-to-month are wrongly flagged 68% of the time (303 of 444), against 33% overall.
2. Customers who stay with PaymentMethod = Electronic check are wrongly flagged 65% of the time (173 of 268), against 33% overall.
3. Customers who stay with tenure_band = 0-12 months are wrongly flagged 65% of the time (151 of 234), against 33% overall.

| feature | value | customers | churners | fn_rate | stayers | fp_rate |
| --- | --- | --- | --- | --- | --- | --- |
| Contract | Month-to-month | 773 | 329 | 0.076 | 444 | 0.682 |
| PaymentMethod | Electronic check | 474 | 206 | 0.073 | 268 | 0.646 |
| tenure_band | 0-12 months | 449 | 215 | 0.088 | 234 | 0.645 |
| InternetService | Fiber optic | 613 | 252 | 0.083 | 361 | 0.609 |
| SeniorCitizen | 1 | 222 | 98 | 0.041 | 124 | 0.573 |
| tenure_band | 13-24 months | 206 | 60 | 0.067 | 146 | 0.493 |
| PaperlessBilling | Yes | 840 | 274 | 0.124 | 566 | 0.440 |
| SeniorCitizen | 0 | 1187 | 276 | 0.203 | 911 | 0.292 |

## Takeaways

- The segments above show where errors concentrate. A segment with a miss rate far above the overall rate is where the model has the least signal.
- Segment-level rates rest on small counts, so treat them as leads to investigate rather than firm conclusions.
- Full per-segment numbers are in `outputs/segment_error_rates.csv` and the ten sampled customers are in `outputs/misclassified_sample.csv`.
