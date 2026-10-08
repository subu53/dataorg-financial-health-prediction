# Financial Health Prediction (data.org challenge on Zindi)

My solution code for the [data.org Financial Health Prediction Challenge](https://zindi.world/competitions/dataorg-financial-health-prediction-challenge) on Zindi. The task: sort small businesses in Eswatini, Zimbabwe, Malawi and Lesotho into Low, Medium or High financial health. The metric is F1. The challenge closed on 15 March 2026.

## The data

- 9,618 businesses in the training set: 6,280 Low, 2,868 Medium, 470 High.
- Money columns are self-reported in each country's local currency. Median annual turnover runs from 1,000 in Lesotho to 600,000 in Malawi.
- The data dictionary defines turnover as annual and expenses as "monthly or annual", so the same business gives expense figures 12 times apart depending on the period the owner picked.
- 860 businesses (9.2% of those with both figures) report expenses above turnover. More than half of the turnover answers are whole thousands.

## What the code does

`fhpbestcode089.py`:

- Fills missing "has_" answers with "Never had" and missing money values with the median of the positive values.
- Builds ratio features (turnover over expenses, profit margin, income over turnover, and others). A ratio cancels the currency, since both figures come from the same owner.
- Target-encodes country and owner sex out of fold, so no row's own label leaks into its feature.
- Adds country-level means and medians, and an 8-cluster KMeans label on the scaled money columns.
- Trains LightGBM, XGBoost and CatBoost with 7-fold stratified CV, blends them with weights from each model's out-of-fold weighted F1, then tunes class cut-offs on the out-of-fold predictions.

## Known limits

- The ratios don't fix the monthly-or-annual expenses problem.
- Blend weights and cut-offs are both tuned on the same out-of-fold predictions, so the out-of-fold F1 runs optimistic.

## Run it

The script reads `/kaggle/input/fhp-challenge/Train (1).csv` and `Test (1).csv`. On Kaggle, upload this repo's `Train (5).csv` and `Test (5).csv` to a dataset named `fhp-challenge` under those names. Anywhere else, edit the two paths in `load_data()`. Then run:

```bash
python fhpbestcode089.py
```

It writes `submission.csv`.

## Licence

- Code: MIT.
- Data: provided by data.org through Zindi under CC-BY-SA 4.0. `Train (5).csv`, `Test (5).csv`, `SampleSubmission (7).csv` and `VariableDefinitions (1).csv` are shared here under the same licence.
