# 🏭 Industrial Predictive Maintenance & Failure Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Best Accuracy](https://img.shields.io/badge/Accuracy-99.15%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../../LICENSE)

## 📌 Executive Overview
In modern **Industry 4.0** and automated manufacturing environments, unexpected machine downtime results in substantial financial losses and operational hazards. This project delivers an end-to-end Machine Learning diagnostic engine capable of:
1. **Early Anomaly Detection**: Proactively identifying imminent machinery breakdown before severe failure occurs.
2. **Multi-Class Root-Cause Diagnostics**: Classifying specific industrial failure mechanisms to guide rapid maintenance response.

---

## 📊 Dataset Architecture & Features
Built upon the standardized **AI4I 2020 Predictive Maintenance Dataset** representing 10,000 synthetic industrial milling machines:

| Feature Name | Type | Description |
|---|---|---|
| `Type` | Categorical | Machine variant quality grade: `Low (L)`, `Medium (M)`, `High (H)` |
| `Air temperature [K]` | Numerical | Ambient operating room temperature |
| `Process temperature [K]` | Numerical | Internal operational process temperature |
| `Rotational speed [rpm]` | Numerical | Spindle rotational speed |
| `Torque [Nm]` | Numerical | Spindle motor torque |
| `Tool wear [min]` | Numerical | Cumulative cutter tool wear time in minutes |
| `Target` | Binary Label | `0` = Normal Operation, `1` = Machine Failure |
| `Failure Type` | Multi-class Label | `No Failure`, `Heat Dissipation Failure (HDF)`, `Power Failure (PWF)`, `Overstrain Failure (OSF)`, `Tool Wear Failure (TWF)`, `Random Failures (RNF)` |

---

## 🔬 Machine Learning Pipeline

```mermaid
flowchart LR
    A["Raw Industrial Sensor Telemetry (10k Rows)"] --> B["Data Cleaning & Encoding (Type L/M/H)"]
    B --> C["Outlier Profiling & IQR Analysis"]
    C --> D["Feature Scaling (StandardScaler)"]
    D --> E["Feature Selection (RF Importance Analysis)"]
    E --> F["Stratified Train/Test Split (80/20)"]
    F --> G["Model Benchmarking (RF, DT, SVM, KNN)"]
    G --> H["Final Diagnosis & Evaluation (99.15% Accuracy)"]
```

### 1. Data Cleaning & Outlier Management
- Non-predictive identification artifacts (`UDI`, `Product ID`) were isolated and removed to avoid spurious correlations.
- Rigorous **Interquartile Range (IQR)** detection identified realistic operational extremes (e.g., high torque spikes during heavy cutting passes).

### 2. Feature Selection & Ablation Study
We conducted comparative ablation studies evaluating:
- **Baseline Feature Set** vs. **Random Forest Feature Importance Filtering** vs. **Dropping Redundant Collinear Features**.
- Feature importance analysis revealed **Torque [Nm]**, **Rotational speed [rpm]**, and **Tool wear [min]** as the dominant drivers of machine failure.

---

## 🏆 Benchmark & Experimental Results

Evaluated across $n = 2,000$ test observations under stratified sampling:

| Model Architecture | Binary Failure Accuracy | Multi-Class Failure Diagnosis Accuracy | Primary Strength |
|---|:---:|:---:|---|
| **Random Forest (100 Trees)** | **99.10%** | **99.15%** | **Robust to non-linear sensor interactions & tool wear thresholds** |
| **Decision Tree (CART)** | 98.45% | 98.60% | High interpretability for industrial safety rules |
| **K-Nearest Neighbors ($k=5$)** | 97.20% | 97.90% | Strong local manifold clustering |
| **Support Vector Machine (RBF)** | 97.75% | 97.05% | Effective high-dimensional margin boundary |

### Key Diagnostic Takeaway
Random Forest demonstrated near-flawless discrimination across both binary and multi-class objectives. Crucially, the model maintained high recall on critical rare failure modes (`Power Failure` and `Heat Dissipation`), preventing catastrophic false negatives in industrial production.

---

## 📂 Project Organization
```
01-predictive-maintenance/
├── README.md                                    # Comprehensive Project Documentation
├── predictive_maintenance_classification.ipynb  # Interactive Exploration & Visualization Notebook
└── src/
    └── train_predictive_maintenance.py          # Modular, CLI-executable training script
```

---

## 🚀 Quickstart & Execution

### 1. Run Interactive Notebook
```bash
jupyter notebook predictive_maintenance_classification.ipynb
```

### 2. Run CLI Training Pipeline
```bash
# Multi-class Failure Type diagnosis
python src/train_predictive_maintenance.py --data_path predictive_maintenance.csv --target_mode multiclass

# Binary Failure prediction
python src/train_predictive_maintenance.py --data_path predictive_maintenance.csv --target_mode binary
```

---
*Developed by Amr Gamal as part of the EME Artificial Intelligence & Machine Learning Program.*
