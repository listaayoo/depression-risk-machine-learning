# Identifying Depression Risk Factors Using Machine Learning

## Overview
This project aims to identify depression risk factors based on socio-demographic and lifestyle variables using machine learning techniques.  
The analysis applies both **unsupervised learning (clustering)** and **supervised learning (classification)** to understand patterns of depression risk and to predict risk levels with high accuracy.

The project was developed as part of an academic study in the **Information Systems Department, Universitas Multimedia Nusantara**.

---

## Objectives
- Identify key socio-demographic and lifestyle factors associated with depression risk
- Group individuals into depression risk categories using clustering
- Predict depression risk levels using classification models
- Visualize factor contributions to better understand depression risk patterns

---

## Dataset
- Source: **Public dataset from Kaggle**
- Format: CSV
- Original size: 413,768 rows and 16 columns
- Sample used: 10,000 rows (for computational efficiency)
- **Note:** The dataset is **not included** in this repository

The dataset contains variables such as:
- Age
- Marital Status
- Education Level
- Income
- Number of Children
- Smoking Status
- Physical Activity Level
- Sleep Patterns
- Alcohol Consumption
- Mental Health History

---

## Methodology

### 1. Data Preprocessing
- Missing value inspection
- Categorical encoding (Label Encoding)
- Binning and grouping
- Normalization using `StandardScaler`
- Outlier detection and handling using IQR method

### 2. Data Visualization
The following visualizations are used to explore data patterns:
- Histogram (Income distribution)
- Bar chart (Smoking status)
- Line plot (Sleep patterns vs age)
- Pie chart (Marital status)
- Heatmap (Correlation between numerical features)
- Spider/Radar chart (Risk factor comparison across clusters)

### 3. Clustering
- Algorithm: **K-Means**
- Purpose: Generate depression risk labels
- Output clusters:
  - Not at Risk
  - At Risk
  - High Risk

### 4. Classification
- Algorithms:
  - **K-Nearest Neighbors (KNN)**
  - **Support Vector Machine (SVM)**
- Validation:
  - Train-test split (70:30)
  - Cross-validation
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## Results

### Model Performance
| Model | Accuracy |
|------|----------|
| KNN  | 95% |
| SVM  | 98% |

- SVM demonstrated more stable and superior performance compared to KNN
- Both models achieved high precision and recall across all risk categories

### Key Findings
- Depression risk is influenced by a **combination of factors**, not a single variable
- Higher education does not always correlate with lower depression risk
- Income, smoking status, physical activity, and marital status play significant roles
- Spider chart visualization highlights increasing imbalance of lifestyle factors as risk level rises

---

## Tools & Technologies
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Project Structure
├── analysis.ipynb
└── README.md

---

## Notes
- This repository is intended for **academic and portfolio purposes**
- Dataset is not included due to licensing and size considerations
- Analysis and results can be reviewed directly through the notebook outputs

---

## Author
**Delista Dwi Widyastuti**  