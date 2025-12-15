import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Overview of Dataset
# -------------------------------


def dataset_overview(df):
    """Return basic overview of the dataset."""
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes,
        "head": df.head(),
        "tail": df.tail()
    }


# -------------------------------
# 2. Summary Statistics
# -------------------------------
def summary_statistics(df):
    """Return descriptive statistics for numeric features."""
    return df.describe()


# -------------------------------
# 3. Missing Values
# -------------------------------
def missing_values(df):
    """Return missing value counts and proportions."""
    null_sums = df.isnull().sum()
    return pd.DataFrame({
        "Column": null_sums.index,
        "Missing Values": null_sums.values,
        "Proportion": null_sums.values / len(df)
    })


# -------------------------------
# 4. Data Types Split
# -------------------------------
def get_numeric_columns(df):
    return df.select_dtypes(include=['int', 'float']).columns.tolist()


def get_categorical_columns(df):
    return df.select_dtypes(include=['object']).columns.tolist()


# -------------------------------
# 5. Distribution Visualization
# -------------------------------
def plot_numeric_distributions(df):
    numeric_cols = get_numeric_columns(df)

    for col in numeric_cols:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.hist(df[col].dropna(), bins=30)
        plt.title(f"Histogram: {col}")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot: {col}")

        plt.show()


def plot_categorical_distributions(df):
    cat_cols = get_categorical_columns(df)

    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=df[col])
        plt.title(f"Count Plot: {col}")
        plt.xticks(rotation=45)
        plt.show()


# -------------------------------
# 6. Correlation Analysis
# -------------------------------
def plot_correlation_heatmap(df):
    numeric_cols = get_numeric_columns(df)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="Blues")
    plt.title("Correlation Heatmap")
    plt.show()


# -------------------------------
# 7. Outlier Detection (IQR) used boxplot to visualize
# -------------------------------
def detect_outliers_iqr(df, column):
    """Return outliers and bounds for any numeric column."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper
