# src/task_5/train_pipeline.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_preprocessing_pipeline():
    """
    Preprocessing pipeline for raw numeric and categorical features.
    """
    numeric_features = ["Amount", "Value"]
    categorical_features = ["CountryCode",
                            "PricingStrategy", "CurrencyCode_UGX"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor
