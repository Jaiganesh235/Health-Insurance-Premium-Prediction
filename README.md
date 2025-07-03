# 🏥 Health Insurance Premium Prediction

A Machine Learning project that predicts individual medical insurance premiums based on demographic and lifestyle factors. Built with Python and Streamlit for an interactive UI.

---

## 📌 Overview

This project leverages regression-based machine learning models to predict health insurance premium charges using features like age, BMI, smoking status, and region. The model helps insurers better assess risk and customers understand premium costs.

---

## 📁 Dataset

- **Source:** insurance.csv  
- **Features:**
  - `age` – Age of the individual
  - `sex` – Gender
  - `bmi` – Body Mass Index
  - `children` – Number of dependents
  - `smoker` – Smoking status
  - `region` – Residential region
  - `charges` – Target variable (insurance premium)

---

## 🎯 Objectives

- Perform exploratory data analysis (EDA) and feature engineering
- Train and compare multiple regression models
- Explain feature importance using SHAP
- Build an interactive UI using **Streamlit** for premium prediction

---

## 🧠 Machine Learning Models

| Model                     | R² Score | MAE       | RMSE      |
|--------------------------|----------|-----------|-----------|
| XGBoost Regression        | 0.875    | 2468.39   | 4276.51   |
| Gradient Boosting         | 0.865    | 2500.26   | 4445.76   |
| Random Forest Regression  | 0.858    | 2523.65   | 4556.86   |
| LightGBM Regression       | 0.857    | 2701.24   | 4579.69   |
| K-Nearest Neighbors       | 0.822    | 3168.48   | 5112.99   |
| Decision Tree Regression  | 0.710    | 3062.05   | 6517.46   |

---

## 🔍 Feature Importance

Using **SHAP (SHapley Additive Explanations)**, we determined that the most influential features include:

- **Smoking status**
- **BMI**
- **Age**

These features significantly impact premium predictions and help in model interpretability.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Scikit-learn, Pandas, NumPy, XGBoost, LightGBM, SHAP, Matplotlib, Seaborn
- **UI:** Streamlit

---

