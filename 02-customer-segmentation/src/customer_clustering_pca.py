"""
Customer Personality Segmentation & Clustering Pipeline
Author: Amr Gamal
Organization: EME Internship Portfolio

This module implements an Unsupervised Machine Learning pipeline for Customer
Personality Analysis and Behavioral Segmentation. It extracts actionable customer
personas using Feature Engineering, K-Means Clustering, Silhouette Analysis,
and 2D Principal Component Analysis (PCA) projection.
"""

import argparse
import logging
import os
import sys
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def engineer_customer_features(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw marketing campaign dataset, cleans anomalies, and engineers
    behavioral, financial, and demographic indicators.
    """
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found at: {filepath}")
        sys.exit(1)

    logger.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep="\t" if "\t" in open(filepath).readline() else ",")
    logger.info(f"Raw shape: {df.shape}")

    # Impute missing Income with median
    if df["Income"].isnull().sum() > 0:
        median_income = df["Income"].median()
        df["Income"].fillna(median_income, inplace=True)
        logger.info(f"Imputed missing Income values with median: ${median_income:,.2f}")

    # Outlier treatment for extreme unrealistic values
    df = df[df["Year_Birth"] > 1920].copy()
    df = df[df["Income"] < 300000].copy()

    # Feature Engineering
    current_year = 2024
    df["Age"] = current_year - df["Year_Birth"]

    spend_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    available_spend = [c for c in spend_cols if c in df.columns]
    df["Total_Spend"] = df[available_spend].sum(axis=1)

    df["Total_Children"] = df.get("Kidhome", 0) + df.get("Teenhome", 0)
    df["Is_Parent"] = (df["Total_Children"] > 0).astype(int)

    purchase_cols = ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    available_purchases = [c for c in purchase_cols if c in df.columns]
    df["Total_Purchases"] = df[available_purchases].sum(axis=1)

    cmp_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]
    available_cmps = [c for c in cmp_cols if c in df.columns]
    df["Total_Accepted_Campaigns"] = df[available_cmps].sum(axis=1)

    # Select numerical features for segmentation
    feature_cols = [
        "Age", "Income", "Total_Spend", "Total_Children", "Is_Parent",
        "Total_Purchases", "Total_Accepted_Campaigns", "NumWebVisitsMonth", "Recency"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    logger.info(f"Engineered feature matrix shape: {X.shape}")
    return df, X


def find_optimal_clusters(X_scaled: np.ndarray, k_range: range = range(2, 10)) -> Dict[int, float]:
    """Computes Inertia and Silhouette scores across a range of k."""
    logger.info("Computing Elbow & Silhouette Scores across cluster range...")
    silhouette_map = {}
    print(f"\n{'k':<4} | {'Inertia':<12} | {'Silhouette Score':<16}")
    print("-" * 38)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        silhouette_map[k] = sil
        print(f"{k:<4} | {km.inertia_:<12.1f} | {sil:<16.4f}")
    return silhouette_map


def profile_clusters(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """Computes mean profile metrics for each assigned cluster."""
    df_clustered = df.copy()
    df_clustered["Cluster"] = cluster_labels

    persona_map = {
        0: "Older, Moderate-Income Steady Spenders",
        1: "Low-Income, Low-Engagement Families",
        2: "High-Income, Luxury Enthusiasts",
        3: "Mid-Career Value Seekers"
    }

    df_clustered["Persona"] = df_clustered["Cluster"].map(lambda c: persona_map.get(c, f"Cluster {c}"))

    summary_cols = ["Age", "Income", "Total_Spend", "Total_Children", "Total_Purchases", "Total_Accepted_Campaigns"]
    summary = df_clustered.groupby("Persona")[summary_cols].mean().round(2)
    summary["Cluster_Size"] = df_clustered["Persona"].value_counts()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Customer Segmentation & Persona Clustering Pipeline")
    parser.add_argument("--data_path", type=str, default="marketing_campaign.csv", help="Path to raw dataset")
    parser.add_argument("--n_clusters", type=int, default=4, help="Optimal number of clusters (default: 4)")
    args = parser.parse_args()

    df, X = engineer_customer_features(args.data_path)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate Cluster Quality
    find_optimal_clusters(X_scaled)

    # Fit Final KMeans Model
    logger.info(f"Fitting final KMeans with k = {args.n_clusters}...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # PCA Projection for 2D Visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_ * 100}%")

    # Profile Summary
    profile_table = profile_clusters(df, cluster_labels)
    print("\n" + "=" * 75)
    print("CUSTOMER SEGMENTATION SUMMARY PROFILES")
    print("=" * 75)
    print(profile_table.to_string())


if __name__ == "__main__":
    main()
