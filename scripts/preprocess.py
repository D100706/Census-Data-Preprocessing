"""
Preprocessing Pipeline: UCI Adult Census Income Dataset

This script:
1. Loads the raw dataset files.
2. Cleans column names and text values.
3. Handles missing values and duplicate rows.
4. Validates data types and outliers.
5. Creates new features.
6. Saves the cleaned dataset, summary report, and charts.

Run from the project root:
    python scripts/preprocess.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"

for folder in [PROCESSED_DIR, OUTPUT_DIR, DOCS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. DATASET COLUMN NAMES
# --------------------------------------------------

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


# --------------------------------------------------
# 3. LOAD LOCAL DATASET FILES
# --------------------------------------------------

def load_dataset():
    """Load and combine the Adult training and test files."""

    train_path = RAW_DIR / "adult.data"
    test_path = RAW_DIR / "adult.test"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {train_path}\n"
            "Please place adult.data inside data/raw."
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {test_path}\n"
            "Please place adult.test inside data/raw."
        )

    print("Loading adult.data...")

    train = pd.read_csv(
        train_path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values=["?", " ?", ""],
        comment="|",
        skip_blank_lines=True,
    )

    print("Loading adult.test...")

    test = pd.read_csv(
        test_path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values=["?", " ?", ""],
        comment="|",
        skip_blank_lines=True,
    )

    # Combine both files.
    df = pd.concat([train, test], ignore_index=True)

    # Remove unnecessary spaces from text columns.
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].str.strip()

    # The test file's income labels end with a period.
    df["income"] = df["income"].str.rstrip(".")

    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Combined dataset shape: {df.shape}")

    return df


# --------------------------------------------------
# 4. DATA CLEANING
# --------------------------------------------------

def clean_dataset(df):
    """Clean missing values, duplicates, and invalid values."""

    summary = {}

    summary["initial_rows"] = int(len(df))
    summary["initial_columns"] = int(len(df.columns))
    summary["initial_missing_values"] = int(df.isna().sum().sum())

    print("\nStarting data cleaning...")

    # Convert numeric columns to appropriate numeric data types.
    numeric_columns = [
        "age",
        "fnlwgt",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Replace impossible or invalid values with missing values.
    df.loc[~df["age"].between(17, 100), "age"] = np.nan

    df.loc[
        ~df["hours_per_week"].between(1, 99),
        "hours_per_week",
    ] = np.nan

    df.loc[df["fnlwgt"] <= 0, "fnlwgt"] = np.nan

    df.loc[df["education_num"] <= 0, "education_num"] = np.nan

    df.loc[df["capital_gain"] < 0, "capital_gain"] = np.nan

    df.loc[df["capital_loss"] < 0, "capital_loss"] = np.nan

    # Record the number of duplicate rows before removing them.
    duplicates_before = int(df.duplicated().sum())

    # Remove exact duplicate records.
    df = df.drop_duplicates().copy()

    summary["duplicate_rows_removed"] = duplicates_before

    # The target variable should not be guessed or imputed.
    # Remove rows without an income label.
    missing_income_rows = int(df["income"].isna().sum())

    df = df.dropna(subset=["income"]).copy()

    summary["rows_removed_missing_income"] = missing_income_rows

    # Fill missing numerical values with the median.
    for column in numeric_columns:
        median_value = df[column].median()

        if pd.notna(median_value):
            df[column] = df[column].fillna(median_value)

    # Fill missing categorical values with "Unknown".
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown")

    # Normalize text values after filling missing entries.
    for column in categorical_columns:
        df[column] = df[column].astype(str).str.strip()

    # Store summary statistics.
    summary["rows_after_cleaning"] = int(len(df))
    summary["columns_after_cleaning"] = int(len(df.columns))
    summary["missing_values_after_cleaning"] = int(
        df.isna().sum().sum()
    )

    print(f"Duplicate rows removed: {summary['duplicate_rows_removed']:,}")
    print(
        "Rows removed for missing income:",
        f"{summary['rows_removed_missing_income']:,}",
    )
    print(f"Rows after cleaning: {len(df):,}")
    print(
        "Missing values remaining:",
        summary["missing_values_after_cleaning"],
    )

    return df, summary


# --------------------------------------------------
# 5. FEATURE ENGINEERING
# --------------------------------------------------

def engineer_features(df):
    """Create useful features from existing columns."""

    print("\nCreating new features...")

    # Age groups.
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-65",
            "66+",
        ],
        include_lowest=True,
    )

    df["age_group"] = df["age_group"].astype(str)

    # Weekly working-hour categories.
    df["work_hours_band"] = pd.cut(
        df["hours_per_week"],
        bins=[0, 20, 40, 60, 100],
        labels=[
            "Part-time",
            "Standard",
            "Overtime",
            "High-hours",
        ],
        include_lowest=True,
    )

    df["work_hours_band"] = df["work_hours_band"].astype(str)

    # Binary indicators for capital gains and losses.
    df["has_capital_gain"] = (
        df["capital_gain"] > 0
    ).astype(int)

    df["has_capital_loss"] = (
        df["capital_loss"] > 0
    ).astype(int)

    # Log transformations reduce the effect of highly skewed values.
    df["capital_gain_log"] = np.log1p(
        df["capital_gain"].clip(lower=0)
    )

    df["capital_loss_log"] = np.log1p(
        df["capital_loss"].clip(lower=0)
    )

    print("Created 6 engineered features.")

    return df


# --------------------------------------------------
# 6. SAVE CLEANED DATASET
# --------------------------------------------------

def save_cleaned_dataset(df):
    """Save the cleaned data as a CSV file."""

    output_path = (
        PROCESSED_DIR / "adult_census_clean.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved to:\n{output_path}")

    return output_path


# --------------------------------------------------
# 7. GENERATE VISUALIZATIONS
# --------------------------------------------------

def create_visualizations(df):
    """Save charts describing the cleaned dataset."""

    sns.set_theme(style="whitegrid")

    # Income distribution.
    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="income",
        order=sorted(df["income"].unique()),
    )

    plt.title("Income Class Distribution")
    plt.xlabel("Income Class")
    plt.ylabel("Number of Records")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "income_distribution.png",
        dpi=300,
    )

    plt.close()

    # Missing values after cleaning.
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    plt.figure(figsize=(10, 5))

    if len(missing_counts) > 0:
        missing_counts.sort_values().plot(kind="barh")
        plt.title("Missing Values After Cleaning")
        plt.xlabel("Number of Missing Values")
    else:
        plt.text(
            0.5,
            0.5,
            "No Missing Values Remaining",
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.title("Missing Values After Cleaning")
        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "missing_values_after.png",
        dpi=300,
    )

    plt.close()

    print("Visualizations saved in the outputs folder.")


# --------------------------------------------------
# 8. SAVE PREPROCESSING SUMMARY
# --------------------------------------------------

def save_summary(summary, df):
    """Save a summary of the preprocessing pipeline."""

    summary["final_rows"] = int(len(df))
    summary["final_columns"] = int(len(df.columns))
    summary["final_missing_values"] = int(
        df.isna().sum().sum()
    )

    summary["final_column_names"] = df.columns.tolist()

    summary["income_distribution"] = {
        str(key): int(value)
        for key, value in df["income"].value_counts().items()
    }

    summary_path = (
        OUTPUT_DIR / "preprocessing_summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Preprocessing summary saved to:\n{summary_path}")


# --------------------------------------------------
# 9. MAIN PIPELINE
# --------------------------------------------------

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("=" * 60)
    print("CENSUS DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load the raw dataset.
    df = load_dataset()

    # Step 2: Clean the data.
    df, summary = clean_dataset(df)

    # Step 3: Create additional features.
    df = engineer_features(df)

    # Step 4: Save the processed dataset.
    save_cleaned_dataset(df)

    # Step 5: Create charts.
    create_visualizations(df)

    # Step 6: Save the summary report.
    save_summary(summary, df)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nFinal dataset shape:", df.shape)

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    print("\nOutput files:")
    print("- data/processed/adult_census_clean.csv")
    print("- outputs/preprocessing_summary.json")
    print("- outputs/income_distribution.png")
    print("- outputs/missing_values_after.png")


if __name__ == "__main__":
    main()