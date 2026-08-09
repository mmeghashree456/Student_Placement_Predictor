# 🎓 Student Placement Predictor

> An end-to-end Machine Learning platform that analyzes student academic performance, technical skills, projects, internships and other career-related factors to estimate placement outcomes.

🌐 **Live Demo:** https://student-placement-predictor.streamlit.app/

---

## 📌 Overview

**Student Placement Predictor** is a machine learning-based web application designed to help students understand their placement readiness.

The system takes a student's academic profile, technical skills and experience as input and uses trained machine learning models to estimate placement outcomes.

Along with the prediction, the application provides profile-level analysis and actionable insights that can help students identify areas for improvement.

---

## ✨ Features

### 🎯 Placement Prediction
Enter a student's profile and generate a placement prediction using trained ML models.

The prediction considers factors such as:

- CGPA
- Backlogs
- Coding skills
- Communication skills
- Soft skills
- Aptitude score
- Projects
- Internships
- Certifications
- Academic background
- Age and other profile information

### 📊 EDA Dashboard

Explore the training dataset through an interactive Exploratory Data Analysis dashboard.

It provides insights into:

- Dataset distributions
- Academic performance
- Skill-related factors
- Placement trends
- Feature relationships

### 📁 Batch Upload

Upload student data in CSV format and process multiple student profiles together.

This makes the system useful beyond individual predictions.

### 🧠 Multiple ML Models

The project includes multiple classification approaches:

- Logistic Regression
- Random Forest
- XGBoost

The trained models are stored and loaded using `joblib`.

### 📈 Profile Analysis

The application analyzes a student's profile and highlights important academic, technical and experience-related factors.

### 🚀 Career Readiness

The system provides a career-readiness perspective based on the student's overall profile and identifies areas that can potentially be improved.

---

## 🏗️ Project Architecture

```text
Student_Placement_Predictor/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   └── processed/
│
├── models/
│   ├── best_classifier.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── classification_results.csv
│   └── feature_importance.csv
│
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── train_models.py
│   └── train_salary.py
│
├── notebooks/
│
├── reports/
│
└── screenshots/
