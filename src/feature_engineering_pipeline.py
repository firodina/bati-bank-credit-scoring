# # -------------------------------------------------------------
# # feature_engineering_pipeline_task3.py
# # Task 3 - Full Feature Engineering Pipeline (Pure Python Version)
# # Adapted for your dataset
# # -------------------------------------------------------------

# import pandas as pd
# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.impute import SimpleImputer

# # -------------------------------------------------------------
# # 1. Aggregate Transaction Features
# # -------------------------------------------------------------


# class AggregateFeatures(BaseEstimator, TransformerMixin):
#     def __init__(self, customer_col, transaction_col):
#         self.customer_col = customer_col
#         self.transaction_col = transaction_col

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         df = X.copy()
#         agg = df.groupby(self.customer_col)[self.transaction_col].agg(
#             Total_Transaction_Amount="sum",
#             Avg_Transaction_Amount="mean",
#             Transaction_Count="count",
#             Std_Transaction_Amount="std"
#         ).reset_index()
#         return df.merge(agg, on=self.customer_col, how="left")

# # -------------------------------------------------------------
# # 2. Datetime Feature Extraction
# # -------------------------------------------------------------


# class TimeFeatures(BaseEstimator, TransformerMixin):
#     def __init__(self, timestamp_col):
#         self.timestamp_col = timestamp_col

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         df = X.copy()
#         df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
#         df["Transaction_Hour"] = df[self.timestamp_col].dt.hour
#         df["Transaction_Day"] = df[self.timestamp_col].dt.day
#         df["Transaction_Month"] = df[self.timestamp_col].dt.month
#         df["Transaction_Year"] = df[self.timestamp_col].dt.year
#         return df.drop(columns=[self.timestamp_col])

# # -------------------------------------------------------------
# # 3. WoE + IV Feature Selection (Custom)
# # -------------------------------------------------------------


# class WOEIVTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, target_col, iv_threshold=0.02, bins=5):
#         self.target_col = target_col
#         self.iv_threshold = iv_threshold
#         self.bins = bins
#         self.woe_maps = {}
#         self.selected_features_ = []
#         self.iv_values_ = None

#     def fit(self, X, y):
#         df = X.copy()
#         df[self.target_col] = y
#         self.iv_values_ = {}

#         for col in df.columns:
#             if col == self.target_col:
#                 continue
#             if df[col].dtype == 'object':
#                 table, iv = self._woe_iv_categorical(df, col)
#             else:
#                 table, iv = self._woe_iv_numeric(df, col)
#             if iv >= self.iv_threshold:
#                 self.selected_features_.append(col)
#                 self.woe_maps[col] = dict(zip(table['Value'], table['WOE']))
#                 self.iv_values_[col] = iv

#         self.iv_values_ = pd.Series(self.iv_values_)
#         return self

#     def transform(self, X):
#         df = X.copy()
#         for col in self.selected_features_:
#             df[col + "_WOE"] = df[col].map(self.woe_maps[col]).fillna(0)
#         return df[[c + "_WOE" for c in self.selected_features_]]

#     def _woe_iv_categorical(self, df, feature):
#         temp = df[[feature, self.target_col]].copy()
#         temp = temp[temp[feature].notna()]
#         temp[feature] = temp[feature].astype(str)
#         grouped = temp.groupby(feature)[self.target_col].agg(['count', 'sum'])
#         grouped.columns = ['total', 'bad']
#         grouped['good'] = grouped['total'] - grouped['bad']
#         grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
#         grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
#         grouped['good_dist'] = grouped['good_dist'].replace(0, 1e-9)
#         grouped['bad_dist'] = grouped['bad_dist'].replace(0, 1e-9)
#         grouped['WOE'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
#         grouped['IV'] = (grouped['good_dist'] -
#                          grouped['bad_dist']) * grouped['WOE']
#         return grouped.reset_index().rename(columns={feature: 'Value'}), grouped['IV'].sum()

#     def _woe_iv_numeric(self, df, feature):
#         df2 = df[[feature, self.target_col]].copy()
#         df2['bin'] = pd.qcut(df2[feature], q=self.bins, duplicates='drop')
#         grouped = df2.groupby('bin')[self.target_col].agg(['count', 'sum'])
#         grouped.columns = ['total', 'bad']
#         grouped['good'] = grouped['total'] - grouped['bad']
#         grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
#         grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
#         grouped['good_dist'] = grouped['good_dist'].replace(0, 1e-9)
#         grouped['bad_dist'] = grouped['bad_dist'].replace(0, 1e-9)
#         grouped['WOE'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
#         grouped['IV'] = (grouped['good_dist'] -
#                          grouped['bad_dist']) * grouped['WOE']
#         grouped_table = pd.DataFrame(
#             {'Value': grouped.index, 'WOE': grouped['WOE']})
#         return grouped_table, grouped['IV'].sum()

# # -------------------------------------------------------------
# # 4. Full Task 3 Pipeline
# # -------------------------------------------------------------


# def build_task3_pipeline(numeric_cols, categorical_cols, customer_col, transaction_col, timestamp_col):
#     numeric_pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler())
#     ])
#     categorical_pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

#     ])
#     preprocessor = ColumnTransformer([
#         ("num", numeric_pipeline, numeric_cols),
#         ("cat", categorical_pipeline, categorical_cols)
#     ])
#     return Pipeline([
#         ("aggregate_features", AggregateFeatures(customer_col, transaction_col)),
#         ("time_features", TimeFeatures(timestamp_col)),
#         ("preprocessor", preprocessor)
#     ])


# # -------------------------------------------------------------
# # 5. Example Usage
# # -------------------------------------------------------------
# if __name__ == "__main__":
#     # Replace with your actual dataset path
#     df = pd.read_csv("transactions.csv")
#     target = "FraudResult"

#     customer_col = "CustomerId"
#     transaction_col = "Amount"
#     timestamp_col = "TransactionStartTime"

#     categorical_cols = df.select_dtypes(include="object").columns.tolist()
#     # Remove ID columns from features
#     for col in ["TransactionId", "BatchId", "AccountId", "SubscriptionId", customer_col]:
#         if col in categorical_cols:
#             categorical_cols.remove(col)

#     numeric_cols = df.select_dtypes(include="number").columns.tolist()
#     numeric_cols.remove(target)

#     # ---- Task 3 pipeline ----
#     pipeline = build_task3_pipeline(
#         numeric_cols, categorical_cols, customer_col, transaction_col, timestamp_col)
#     X = df.drop(columns=[target])
#     y = df[target]
#     X_task3 = pipeline.fit_transform(X, y)

#     # ---- WoE + IV ----
#     woe_iv = WOEIVTransformer(target_col=target, iv_threshold=0.02)
#     X_woe_iv = woe_iv.fit_transform(df[categorical_cols + numeric_cols], y)

#     print("Task 3 Feature Engineering Completed Successfully")
#     print("Main Feature Matrix Shape:", X_task3.shape)
#     print("WoE-IV Feature Matrix Shape:", X_woe_iv.shape)
#     print("\nInformation Value (IV):")
#     print(woe_iv.iv_values_.sort_values(ascending=False))

# -------------------------------------------------------------
# feature_engineering_pipeline_task3.py
# Task 3 - Full Feature Engineering Pipeline (Pure Python Version)
# Adapted for your dataset, now outputs a new DataFrame
# -------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# -------------------------------------------------------------
# 1. Aggregate Transaction Features
# -------------------------------------------------------------


class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, customer_col, transaction_col):
        self.customer_col = customer_col
        self.transaction_col = transaction_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        agg = df.groupby(self.customer_col)[self.transaction_col].agg(
            Total_Transaction_Amount="sum",
            Avg_Transaction_Amount="mean",
            Transaction_Count="count",
            Std_Transaction_Amount="std"
        ).reset_index()
        return df.merge(agg, on=self.customer_col, how="left")

# -------------------------------------------------------------
# 2. Datetime Feature Extraction
# -------------------------------------------------------------


class TimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, timestamp_col):
        self.timestamp_col = timestamp_col  # store column name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df["Transaction_Hour"] = df[self.timestamp_col].dt.hour
        df["Transaction_Day"] = df[self.timestamp_col].dt.day
        df["Transaction_Month"] = df[self.timestamp_col].dt.month
        df["Transaction_Year"] = df[self.timestamp_col].dt.year
        return df  # keep timestamp column for Task 4


# -------------------------------------------------------------
# 3. WoE + IV Feature Selection (Custom)
# -------------------------------------------------------------


class WOEIVTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, iv_threshold=0.02, bins=5):
        self.target_col = target_col
        self.iv_threshold = iv_threshold
        self.bins = bins
        self.woe_maps = {}
        self.selected_features_ = []
        self.iv_values_ = None

    def fit(self, X, y):
        df = X.copy()
        df[self.target_col] = y
        self.iv_values_ = {}

        for col in df.columns:
            if col == self.target_col:
                continue
            if df[col].dtype == 'object':
                table, iv = self._woe_iv_categorical(df, col)
            else:
                table, iv = self._woe_iv_numeric(df, col)
            if iv >= self.iv_threshold:
                self.selected_features_.append(col)
                self.woe_maps[col] = dict(zip(table['Value'], table['WOE']))
                self.iv_values_[col] = iv

        self.iv_values_ = pd.Series(self.iv_values_)
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.selected_features_:
            df[col + "_WOE"] = df[col].map(self.woe_maps[col]).fillna(0)
        return df[[c + "_WOE" for c in self.selected_features_]]

    def _woe_iv_categorical(self, df, feature):
        temp = df[[feature, self.target_col]].copy()
        temp = temp[temp[feature].notna()]
        temp[feature] = temp[feature].astype(str)
        grouped = temp.groupby(feature)[self.target_col].agg(['count', 'sum'])
        grouped.columns = ['total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']
        grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
        grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
        grouped['good_dist'] = grouped['good_dist'].replace(0, 1e-9)
        grouped['bad_dist'] = grouped['bad_dist'].replace(0, 1e-9)
        grouped['WOE'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
        grouped['IV'] = (grouped['good_dist'] -
                         grouped['bad_dist']) * grouped['WOE']
        return grouped.reset_index().rename(columns={feature: 'Value'}), grouped['IV'].sum()

    def _woe_iv_numeric(self, df, feature):
        df2 = df[[feature, self.target_col]].copy()
        df2['bin'] = pd.qcut(df2[feature], q=self.bins, duplicates='drop')
        grouped = df2.groupby('bin')[self.target_col].agg(['count', 'sum'])
        grouped.columns = ['total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']
        grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
        grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
        grouped['good_dist'] = grouped['good_dist'].replace(0, 1e-9)
        grouped['bad_dist'] = grouped['bad_dist'].replace(0, 1e-9)
        grouped['WOE'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
        grouped['IV'] = (grouped['good_dist'] -
                         grouped['bad_dist']) * grouped['WOE']
        grouped_table = pd.DataFrame(
            {'Value': grouped.index, 'WOE': grouped['WOE']})
        return grouped_table, grouped['IV'].sum()

# -------------------------------------------------------------
# 4. Full Task 3 Pipeline with Output DataFrame
# -------------------------------------------------------------


def build_task3_pipeline(numeric_cols, categorical_cols, customer_col, transaction_col, timestamp_col):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])
    pipeline = Pipeline([
        ("aggregate_features", AggregateFeatures(customer_col, transaction_col)),
        ("time_features", TimeFeatures(timestamp_col)),
        ("preprocessor", preprocessor)
    ])
    return pipeline


def transform_to_dataframe(pipeline, df, target, numeric_cols, categorical_cols):
    """
    Applies the pipeline and returns a full DataFrame including features and target.
    """
    X = df.drop(columns=[target])
    y = df[target].reset_index(drop=True)

    # Fit-transform the pipeline
    X_transformed = pipeline.fit_transform(X, y)

    # Get column names
    ohe_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat']\
        .named_steps['onehot'].get_feature_names_out(categorical_cols)
    agg_time_features = [
        "Total_Transaction_Amount", "Avg_Transaction_Amount",
        "Transaction_Count", "Std_Transaction_Amount",
        "Transaction_Hour", "Transaction_Day",
        "Transaction_Month", "Transaction_Year"
    ]
    feature_names = agg_time_features + numeric_cols + list(ohe_cols)

    # Convert to DataFrame
    df_task3 = pd.DataFrame(X_transformed, columns=feature_names)
    df_task3[target] = y

    return df_task3


# -------------------------------------------------------------
# 5. Example Usage
# -------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("data/processed/augmented_raw_data_for_task3.csv")
    target = "FraudResult"

    customer_col = "CustomerId"
    transaction_col = "Amount"
    timestamp_col = "TransactionStartTime"
categorical_cols = df.select_dtypes(include="object").columns.tolist()

# Remove ID-like columns
for col in [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    customer_col,
    timestamp_col  # ✅ VERY IMPORTANT
]:
    if col in categorical_cols:
        categorical_cols.remove(col)

numeric_cols = df.select_dtypes(include="number").columns.tolist()

# Remove target
if target in numeric_cols:
    numeric_cols.remove(target)

    # -------------------------------
    # 1. Build and transform main pipeline
    # -------------------------------
    pipeline = build_task3_pipeline(
        numeric_cols, categorical_cols, customer_col, transaction_col, timestamp_col)
    X = df.drop(columns=[target])
    y = df[target]

    X_task3 = pipeline.fit_transform(X, y)

    # Convert pipeline output to DataFrame with proper column names
    numeric_feature_names = numeric_cols
    categorical_feature_names = pipeline.named_steps['preprocessor'].transformers_[
        1][1].named_steps['onehot'].get_feature_names_out(categorical_cols)
    all_feature_names = list(numeric_feature_names) + \
        list(categorical_feature_names)
    df_task3 = pd.DataFrame(X_task3, columns=all_feature_names)
    df_task3[target] = y.values  # Add target column back

    # -------------------------------
    # 2. WoE-IV Features
    # -------------------------------
    woe_iv = WOEIVTransformer(target_col=target, iv_threshold=0.02)
    X_woe_iv = woe_iv.fit_transform(df[categorical_cols + numeric_cols], y)
    df_woe_iv = pd.DataFrame(X_woe_iv, columns=X_woe_iv.columns)
    df_woe_iv[target] = y.values  # Add target back

    # -------------------------------
    # 3. Merge main pipeline + WoE-IV features
    # -------------------------------
    df_final_task3 = pd.concat([df_task3.reset_index(drop=True), df_woe_iv.drop(
        columns=[target]).reset_index(drop=True)], axis=1)

    print("Final Task 3 dataset shape:", df_final_task3.shape)
    print(df_final_task3.head())

    # -------------------------------
    # 4. Save clean dataset to processed folder
    # -------------------------------
    output_path = r"C:\Users\hp\Desktop\AI projects\bati-bank-credit-scoring\data\processed\clean_data.csv"
    df_final_task3.to_csv(output_path, index=False)
    print(
        f"Task 3 final clean dataset with WoE-IV features saved at: {output_path}")
