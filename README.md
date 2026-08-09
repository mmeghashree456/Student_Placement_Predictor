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

Student Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Train Multiple ML Models
      │
      ├── Logistic Regression
      ├── Random Forest
      └── XGBoost
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Model Serialization
      │
      ▼
Streamlit Application
      │
      ▼
Placement Prediction
      │
      ▼
Profile Analysis + Career Insights


```🛠️ Tech Stack```
Programming Language
Python
Machine Learning
Scikit-learn
XGBoost
Pandas
NumPy
Data Visualization
Matplotlib
Seaborn
Web Application
Streamlit
Model Persistence
Joblib
Development Tools
VS Code
Git
GitHub

```📂 Dataset```

The project uses a student placement dataset containing academic, technical and career-related attributes.

Important Features
Feature	Description
Gender	Student gender
Degree	Educational degree
Branch	Engineering branch
Age	Student age
CGPA	Academic performance
Backlogs	Number of academic backlogs
Internships	Internship experience
Projects	Number / level of projects
Certifications	Relevant certifications
Coding Skills	Programming / technical skill level
Communication Skills	Communication ability
Soft Skills Rating	Overall soft-skill rating
Aptitude Test Score	Aptitude assessment score
Placement	Placement outcome

The dataset is divided into training and testing data.

data/
├── raw/
│   ├── train.csv
│   └── test.csv
│
└── processed/

```⚙️ How to Run Locally```
1. Clone the Repository
git clone https://github.com/mmeghashree456/Student_Placement_Predictor.git
2. Navigate to the Project
cd Student_Placement_Predictor
3. Create a Virtual Environment
python -m venv venv
4. Activate the Virtual Environment

Windows:

venv\Scripts\activate

macOS / Linux:

source venv/bin/activate
5. Install Required Dependencies
pip install -r requirements.txt
6. Run the Streamlit Application
streamlit run app.py

The application will open in your browser at:

http://localhost:8501

```⭐ Support```

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub!
