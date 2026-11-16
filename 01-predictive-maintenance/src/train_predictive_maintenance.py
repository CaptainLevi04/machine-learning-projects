"""
Predictive Maintenance Classification Pipeline
Author: Amr Gamal
Organization: EME Internship Portfolio

This module implements an end-to-end Machine Learning pipeline for Industrial
Equipment Failure Prediction using the AI4I 2020 Predictive Maintenance Dataset.
It supports both binary classification (Failure vs Non-Failure) and multi-class
classification (identifying specific failure modes: Tool Wear, Heat Dissipation,
Power Failure, Overstrain, etc.).
"""

import argparse
import logging
import os
import sys
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_and_preprocess_data(
    filepath: str,
    target_type: str = "multiclass"
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Loads raw dataset, cleans metadata columns, encodes categorical features,
    and prepares feature matrix X and target vector y.
    """
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found at {filepath}")
        sys.exit(1)

    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Raw dataset shape: {df.shape}")

    # Drop non-predictive identifier columns
    drop_cols = [c for c in ["UDI", "Product ID"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Encode Machine Type ('L', 'M', 'H')
    if "Type" in df.columns:
        type_le = LabelEncoder()
        df["Type"] = type_le.fit_transform(df["Type"])
        logger.info(f"Encoded 'Type' categories: {dict(zip(type_le.classes_, range(len(type_le.classes_))))}")

    # Target Selection
    target_le = LabelEncoder()
    if target_type == "binary":
        if "Failure Type" in df.columns:
            df = df.drop(columns=["Failure Type"])
        y = df["Target"]
        X = df.drop(columns=["Target"])
        logger.info(f"Target selected: Binary Failure (Class counts: {np.bincount(y)})")
    else:
        # Multi-class: Failure Type
        if "Target" in df.columns:
            df = df.drop(columns=["Target"])
        y_raw = df["Failure Type"]
        y = target_le.fit_transform(y_raw)
        X = df.drop(columns=["Failure Type"])
        logger.info(f"Target selected: Multi-class Failure Types: {list(target_le.classes_)}")

    return X, y, target_le


def detect_outliers_iqr(df: pd.DataFrame, num_cols: list) -> Dict[str, int]:
    """Calculates IQR outlier statistics for numerical attributes."""
    outlier_counts = {}
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
    return outlier_counts


def train_and_benchmark(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_names: list = None
) -> Dict[str, Any]:
    """Trains and benchmarks four candidate classification algorithms."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel="rbf", random_state=42)
    }

    results = {}
    logger.info("=" * 65)
    logger.info("BENCHMARKING CLASSIFICATION MODELS")
    logger.info("=" * 65)

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "report": classification_report(y_test, preds, target_names=target_names, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, preds)
        }
        logger.info(f"--> {name} Accuracy: {acc * 100:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Predictive Maintenance Classification Trainer")
    parser.add_argument("--data_path", type=str, default="predictive_maintenance.csv", help="Path to CSV dataset")
    parser.add_argument("--target_mode", type=str, choices=["binary", "multiclass"], default="multiclass",
                        help="Prediction task mode: binary or multiclass")
    parser.add_argument("--test_size", type=float, default=0.2, help="Validation/Test set ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Seed for reproducibility")
    args = parser.parse_args()

    X, y, target_le = load_and_preprocess_data(args.data_path, target_type=args.target_mode)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    logger.info(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    target_labels = [str(c) for c in target_le.classes_] if args.target_mode == "multiclass" else ["No Failure", "Failure"]
    results = train_and_benchmark(X_train_scaled, X_test_scaled, y_train, y_test, target_names=target_labels)

    best_model_name = max(results, key=lambda k: results[k]["accuracy"])
    logger.info("=" * 65)
    logger.info(f"TOP PERFORMER: {best_model_name} ({results[best_model_name]['accuracy'] * 100:.2f}%)")
    logger.info("=" * 65)
    print("\nDetailed Classification Report for Top Model:\n")
    print(results[best_model_name]["report"])


if __name__ == "__main__":
    main()
