# Applied Machine Learning 

A curated collection of end-to-end Machine Learning systems, demonstrating practical expertise in statistical data preprocessing, exploratory data analysis, feature engineering, and rigorous benchmarking across industrial, marketing, and educational domains.

Developed during the **EME Training Program** by **Amr Gamal**.

---

## 🚀 Repository Highlights

- **Predictive Maintenance Diagnostics**: Achieved **99.15% classification accuracy** using Random Forest on real-time industrial telemetry for early equipment failure detection.
- **Customer Segmentation & Persona Profiling**: Unsupervised behavioral clustering via K-Means and PCA, extracting 4 distinct commercial personas with actionable marketing strategies.
- **Academic Performance Forecasting**: Supervised continuous regression modeling on 50,000 student records, reaching **73.31% tolerance accuracy** within ±5 grade points.

---

## 📂 Project Catalog

| Directory | Problem Type | Domain | Dataset Size | Primary Model | Benchmark Result |
|---|---|---|---|---|---|
| [`01-predictive-maintenance`](./01-predictive-maintenance) | Binary & Multi-Class Classification | Industrial IoT / Industry 4.0 | 10,000 machine cycles | Random Forest | **99.15% Accuracy** |
| [`02-customer-segmentation`](./02-customer-segmentation) | Unsupervised Clustering & PCA | E-Commerce / Retail Marketing | 2,240 customer profiles | K-Means ($k=4$) | **4 Distinct Personas** |
| [`03-student-performance-prediction`](./03-student-performance-prediction) | Continuous Regression | EdTech & Academic Analytics | 50,000 student records | Linear Regression / RF | **MAE 3.60 / 73.31% Tolerance** |

---

## 🛠️ Tech Stack & Methodologies

- **Languages & Frameworks**: Python 3.9+, Scikit-Learn, Pandas, NumPy, SciPy
- **Data Visualization**: Matplotlib, Seaborn
- **Core Techniques**:
  - Outlier detection with Interquartile Ranges (IQR)
  - Feature transformation (StandardScaler, One-Hot & Label Encoding)
  - Multicollinearity resolution via correlation matrices
  - Random Forest feature importance scoring
  - Elbow method (Inertia) & Silhouette coefficient optimization
  - 2D dimensionality reduction via Principal Component Analysis (PCA)
  - Stratified data partitioning and multi-metric evaluation (Accuracy, F1, Precision, Recall, MAE, RMSE)

---

## 📁 Repository Layout

```
machine-learning-projects/
├── 01-predictive-maintenance/
│   ├── README.md                                    # In-depth industrial problem documentation
│   ├── predictive_maintenance_classification.ipynb  # Interactive data exploration & modeling
│   └── src/
│       └── train_predictive_maintenance.py          # Production training pipeline
├── 02-customer-segmentation/
│   ├── README.md                                    # Customer persona & clustering documentation
│   ├── customer_personality_clustering.ipynb        # Segmentation analysis & PCA visualizations
│   └── src/
│       └── customer_clustering_pca.py               # Automated clustering & profiling script
├── 03-student-performance-prediction/
│   ├── README.md                                    # Educational regression documentation
│   ├── student_grade_regression.ipynb               # Modeling notebook & tolerance metrics
│   └── src/
│       └── train_student_regression.py              # CLI regression pipeline
├── .gitignore                                       # Python & Jupyter ignore rules
├── requirements.txt                                 # Pinned dependencies
└── README.md                                        # Portfolio overview & guide
```

---

## 💻 Environment Setup & Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/CaptainLevi04/machine-learning-projects.git
cd machine-learning-projects
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install required libraries
```bash
pip install -r requirements.txt
```

### 4. Run any project pipeline
```bash
# Example: Run predictive maintenance multi-class trainer
python 01-predictive-maintenance/src/train_predictive_maintenance.py --target_mode multiclass
```

---

## 👤 Author & Acknowledgments

- **Engineer**: Amr Gamal ([@CaptainLevi04](https://github.com/CaptainLevi04))
- **Program**: EME Artificial Intelligence & Machine Learning Training Program
