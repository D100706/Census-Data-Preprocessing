# Census Data Preprocessing and Cleaning Pipeline

A Python-based data preprocessing project using the **UCI Adult Census Income Dataset**. This project demonstrates data cleaning, missing-value handling, duplicate removal, data validation, feature engineering, and data visualization.

## Project Overview

Raw datasets often contain missing values, duplicate records, inconsistent text formatting, and invalid numerical values. These issues can affect the quality of data analysis and machine learning models.

This project implements a reproducible preprocessing pipeline that combines the Adult Census training and test files, cleans the data, creates additional features, and generates useful reports and visualizations.

## Dataset Information

* **Dataset:** Adult Census Income Dataset
* **Source:** UCI Machine Learning Repository
* **Original records:** 48,842
* **Original columns:** 15
* **Records after preprocessing:** 48,790
* **Columns after feature engineering:** 21
* **Duplicate records removed:** 52

The dataset contains demographic and employment-related information, including age, education, occupation, working hours, and income category.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Visual Studio Code

## Project Structure

```text
Census_Data_Preprocessing/
├── data/
│   ├── raw/
│   │   ├── adult.data
│   │   └── adult.test
│   └── processed/
│       └── adult_census_clean.csv
├── docs/
│   ├── data_documentation.md
│   └── DATA_DICTIONARY.md
├── outputs/
│   ├── income_distribution.png
│   ├── missing_values_after.png
│   └── preprocessing_summary.json
├── screenshots/
├── scripts/
│   └── preprocess.py
├── .gitignore
├── README.md
└── requirements.txt
```

**Note:** The `.venv/` directory contains the local Python virtual environment. It is excluded from version control and should not be uploaded to GitHub.

## Data Preprocessing Steps

### 1. Data Loading

Loads the training and test files and combines them into a single dataset.

### 2. Data Cleaning

* Removes unnecessary spaces from text values.
* Converts numerical columns to appropriate numeric types.
* Identifies invalid values in selected numerical columns.
* Removes duplicate records.
* Removes rows with missing income labels.
* Fills missing numerical values using median imputation.
* Replaces missing categorical values with `Unknown`.

### 3. Feature Engineering

Six new features are created:

| Feature            | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `age_group`        | Groups individuals by age.                           |
| `work_hours_band`  | Categorizes weekly working hours.                    |
| `has_capital_gain` | Indicates whether capital gain is greater than zero. |
| `has_capital_loss` | Indicates whether capital loss is greater than zero. |
| `capital_gain_log` | Log-transformed capital gain.                        |
| `capital_loss_log` | Log-transformed capital loss.                        |

### 4. Data Visualization

Generates charts showing income distribution and the missing-value status after preprocessing.

### 5. Summary Report

Saves preprocessing statistics in JSON format.

## Results

| Metric                    | Result |
| ------------------------- | -----: |
| Initial records           | 48,842 |
| Duplicate records removed |     52 |
| Final records             | 48,790 |
| Initial columns           |     15 |
| Final columns             |     21 |
| Missing values remaining  |      0 |

## Generated Outputs

* `data/processed/adult_census_clean.csv` — cleaned dataset.
* `outputs/preprocessing_summary.json` — preprocessing statistics.
* `outputs/income_distribution.png` — income distribution chart.
* `outputs/missing_values_after.png` — missing-value chart.

## How to Run the Project

### Prerequisites

Install Python and Git. Ensure the raw dataset files are available in `data/raw/`.

### 1. Clone the Repository

```bash
git clone https://github.com/D100706/Census-Data-Preprocessing.git
cd Census-Data-Preprocessing
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Preprocessing Script

```bash
python scripts/preprocess.py
```

The cleaned dataset, JSON summary, and visualization files will be generated automatically.

## Documentation

Refer to the files in `docs/` for additional information about the dataset, preprocessing operations, engineered features, and limitations.

## Conclusion

This project demonstrates a structured and reproducible approach to preparing real-world data for analysis. The resulting dataset can be used for exploratory data analysis and further machine learning experiments.

## Author

Developed as a data preprocessing and cleaning project using Python.
