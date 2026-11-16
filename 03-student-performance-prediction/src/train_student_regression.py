"""
Student Academic Performance Regression Pipeline
Author: Amr Gamal
Organization: EME Internship Portfolio

This module implements an end-to-end Machine Learning Regression system for predicting
student academic grade outcomes based on demographic factors, parental educational
background, school curricula, and student progress metrics using the Egypt Education Dataset.
"""

import argparse
import logging
import os
import sys
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def preprocess_education_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Cleans raw education records, encodes curricula types, constructs parental
    education aggregates and mismatch flags.
    """
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found at {filepath}")
        sys.exit(1)

    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Raw dataset shape: {df.shape}")

    target_column = "Grade"
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing from dataset.")

    # Feature Engineering
    if "Father Degree" in df.columns and "Mother Degree" in df.columns:
        higher_ed = ["Master", "PhD", "Bachelor"]
        df["Both_Parents_Higher_Ed"] = (
            df["Father Degree"].isin(higher_ed) & df["Mother Degree"].isin(higher_ed)
        ).astype(int)

    # Convert Curriculum / EduType dummy variables if present
    edutype_cols = [c for c in df.columns if c.startswith("EduType_")]
    if edutype_cols:
        df[edutype_cols] = df[edutype_cols].astype(int)

    # Drop redundant parental columns after feature synthesis
    drop_candidates = ["Father Degree", "Mother Degree", "Student Age"]
    drop_cols = [c for c in drop_candidates if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Separate feature matrix X and target y
    X = df.drop(columns=[target_column])
    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])
    y = df[target_column]

    logger.info(f"Preprocessed feature matrix shape: {X.shape} | Features: {list(X.columns)}")
    return X, y


def evaluate_regression_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float = 5.0
) -> Dict[str, float]:
    """Computes MAE, RMSE, R2, and domain-specific Tolerance Accuracy."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    within_tol = np.abs(y_true - y_pred) <= tolerance
    tolerance_acc = np.mean(within_tol) * 100.0

    logger.info("-" * 55)
    logger.info(f"EVALUATION: {model_name}")
    logger.info(f"Mean Absolute Error (MAE)       : {mae:.2f}")
    logger.info(f"Root Mean Squared Error (RMSE)   : {rmse:.2f}")
    logger.info(f"R-squared (R2 Score)            : {r2:.4f}")
    logger.info(f"Tolerance Accuracy (±{tolerance:g} pts)     : {tolerance_acc:.2f}%")
    logger.info("-" * 55)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Tolerance_Accuracy": tolerance_acc
    }


def main():
    parser = argparse.ArgumentParser(description="Student Grade Regression Pipeline")
    parser.add_argument("--data_path", type=str, default="egypt_education_dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction (default: 0.2)")
    parser.add_argument("--tolerance", type=float, default=5.0, help="Acceptable error margin in grade points")
    args = parser.parse_args()

    X, y = preprocess_education_data(args.data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    logger.info(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    evaluate_regression_model("Linear Regression", y_test, lr_preds, tolerance=args.tolerance)

    # 2. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    evaluate_regression_model("Random Forest Regressor", y_test, rf_preds, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
