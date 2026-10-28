# Census Income Dataset — Data Documentation

## 1. Project Overview

This project preprocesses the UCI Adult Census Income dataset. The objective is to clean raw data, handle missing values, remove duplicate records, validate data types, and create useful features for further data analysis.

## 2. Dataset Information

* **Dataset name:** Adult Census Income Dataset
* **Source:** UCI Machine Learning Repository
* **Raw files:** `adult.data` and `adult.test`
* **Original combined records:** 48,842
* **Original attributes:** 15
* **Records after preprocessing:** 48,790
* **Attributes after feature engineering:** 21
* **Duplicate records removed:** 52

## 3. Data Cleaning Operations

The preprocessing pipeline performs the following operations:

1. Loads and combines the training and test dataset files.
2. Removes unnecessary spaces from text values.
3. Converts numerical columns to numeric data types.
4. Identifies invalid values in selected numerical columns.
5. Removes duplicate records.
6. Removes records without an income label.
7. Fills missing numerical values using median imputation.
8. Fills missing categorical values with `Unknown`.
9. Verifies that no missing values remain.

## 4. Feature Engineering

Six additional features are created:

| Feature            | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `age_group`        | Categorizes individuals into age groups.             |
| `work_hours_band`  | Categorizes weekly working hours.                    |
| `has_capital_gain` | Indicates whether capital gain is greater than zero. |
| `has_capital_loss` | Indicates whether capital loss is greater than zero. |
| `capital_gain_log` | Log-transformed capital gain.                        |
| `capital_loss_log` | Log-transformed capital loss.                        |

## 5. Output Files

* `data/processed/adult_census_clean.csv` — cleaned dataset.
* `outputs/preprocessing_summary.json` — preprocessing statistics.
* `outputs/income_distribution.png` — income distribution visualization.
* `outputs/missing_values_after.png` — missing-value visualization.

## 6. Limitations

The pipeline uses predefined rules to identify invalid numerical values. Median imputation may affect the distribution of numerical variables. Categorical missing values are represented as `Unknown`, which may influence downstream analysis.

## 7. Conclusion

The preprocessing pipeline produced a dataset containing 48,790 records and 21 columns, with no missing values remaining. The cleaned dataset and generated reports can be used for further exploratory data analysis and machine learning experiments.
