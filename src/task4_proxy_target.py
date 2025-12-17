# -------------------------------------------------------------
# task4_proxy_target.py
# Task 4 - Proxy Target Variable Engineering
# -------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def create_high_risk_target(df, customer_col='CustomerId', amount_col='Amount',
                            transaction_col='TransactionStartTime', n_clusters=3,
                            random_state=42, verbose=True):
    """
    Creates a proxy high-risk target variable using RFM and K-Means clustering.

    Parameters:
    -----------
    df : pandas.DataFrame
        The transactions dataset.
    customer_col : str
        Column name for customer ID.
    amount_col : str
        Column name for transaction amount.
    transaction_col : str
        Column name for transaction timestamp.
    n_clusters : int
        Number of clusters for K-Means segmentation.
    random_state : int
        Random state for reproducibility.
    verbose : bool
        If True, prints cluster summary and explanations.

    Returns:
    --------
    df : pandas.DataFrame
        Original dataframe with a new column 'is_high_risk'.
    rfm : pandas.DataFrame
        RFM table with cluster assignments and high-risk label.
    """

    # -------------------------------
    # Step 1: Convert timestamp and define snapshot
    # -------------------------------
    df[transaction_col] = pd.to_datetime(df[transaction_col])
    snapshot_date = df[transaction_col].max() + pd.Timedelta(days=1)

    # -------------------------------
    # Step 2: Calculate RFM metrics
    # -------------------------------
    rfm = df.groupby(customer_col).agg({
        transaction_col: lambda x: (snapshot_date - x.max()).days,  # Recency
        'TransactionId': 'count',  # Frequency
        amount_col: 'sum'  # Monetary
    }).reset_index()

    rfm.rename(columns={
        transaction_col: 'Recency',
        'TransactionId': 'Frequency',
        amount_col: 'Monetary'
    }, inplace=True)

    if verbose:
        print("RFM Table (first 5 rows):\n", rfm.head())

    # -------------------------------
    # Step 3: Scale RFM features
    # -------------------------------
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[['Recency', 'Frequency', 'Monetary']])

    # -------------------------------
    # Step 4: K-Means Clustering
    # -------------------------------
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    if verbose:
        print("\nCluster centroids (scaled RFM features):\n",
              kmeans.cluster_centers_)
        cluster_summary = rfm.groupby(
            'Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        print("\nCluster summary (mean RFM values):\n", cluster_summary)

    # -------------------------------
    # Step 5: Identify high-risk cluster
    # -------------------------------
    cluster_summary = rfm.groupby(
        'Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    # Criteria: high Recency, low Frequency, low Monetary
    high_risk_cluster = cluster_summary.sort_values(
        by=['Recency', 'Frequency', 'Monetary'],
        ascending=[False, True, True]
    ).index[0]

    # Assign binary target
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)

    if verbose:
        print(f"\nHigh-risk cluster identified: {high_risk_cluster}")
        print(rfm[['CustomerId', 'Cluster', 'is_high_risk']].head())

    # -------------------------------
    # Step 6: Merge high-risk target back to original dataset
    # -------------------------------
    df = df.merge(rfm[[customer_col, 'is_high_risk']],
                  on=customer_col, how='left')

    return df, rfm
