import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# -----------------------------
# Load dataset
# -----------------------------

def load_data(path):
    return pd.read_csv(path)


# -----------------------------
# Prepare features and target
# -----------------------------

def prepare_data(df):

    # Remove ID because it has no predictive meaning
    df = df.drop(columns=["Student_ID"])

    # Target
    X = df.drop(columns=["Placement_Status"])
    y = df["Placement_Status"].map({
        "Not Placed": 0,
        "Placed": 1
    })

    return X, y


# -----------------------------
# Feature types
# -----------------------------

CATEGORICAL_FEATURES = [
    "Gender",
    "Degree",
    "Branch"
]

NUMERICAL_FEATURES = [
    "Age",
    "CGPA",
    "Internships",
    "Projects",
    "Coding_Skills",
    "Communication_Skills",
    "Aptitude_Test_Score",
    "Soft_Skills_Rating",
    "Certifications",
    "Backlogs"
]


# -----------------------------
# Preprocessing pipeline
# -----------------------------

def create_preprocessor():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES
            )
        ]
    )

    return preprocessor