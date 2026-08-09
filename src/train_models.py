import os
import joblib
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.data_processing import (
    load_data,
    prepare_data,
    create_preprocessor,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/train.csv"
MODEL_DIR = "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("AI STUDENT PLACEMENT PREDICTOR")
print("=" * 60)

print("\n[1/6] Loading dataset...")

df = load_data(DATA_PATH)

print(f"Dataset shape: {df.shape}")

X, y = prepare_data(df)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

print("\n[2/6] Creating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print("\nClass distribution:")

print(
    pd.Series(y_train)
    .value_counts(normalize=True)
    .sort_index()
)


# ============================================================
# 3. DEFINE MODELS
# ============================================================

print("\n[3/6] Initializing models...")

models = {

    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "xgboost": XGBClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    ),
}


# ============================================================
# 4. TRAIN + EVALUATE
# ============================================================

print("\n[4/6] Training models...")

os.makedirs(MODEL_DIR, exist_ok=True)

results = []

trained_models = {}


for name, model in models.items():

    print("\n" + "-" * 60)
    print(f"Training: {name}")
    print("-" * 60)

    # Create a fresh preprocessor for every model.
    preprocessor = create_preprocessor()

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipeline.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = pipeline.predict(X_test)

    # --------------------------------------------------------
    # Probability predictions
    # --------------------------------------------------------

    if hasattr(pipeline, "predict_proba"):

        y_probability = pipeline.predict_proba(X_test)[:, 1]

    else:

        y_probability = None

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    if y_probability is not None:

        roc_auc = roc_auc_score(
            y_test,
            y_probability,
        )

    else:

        roc_auc = None

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred,
        )
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
        }
    )

    trained_models[name] = pipeline

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"{name}.joblib",
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    print(f"\nSaved model → {model_path}")


# ============================================================
# 5. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("[5/6] MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

# F1 is used instead of accuracy alone because it balances
# precision and recall.

best_model_name = (
    results_df
    .sort_values(
        by="F1",
        ascending=False,
    )
    .iloc[0]["Model"]
)

best_model = trained_models[best_model_name]

best_model_path = os.path.join(
    MODEL_DIR,
    "best_classifier.joblib",
)

joblib.dump(
    best_model,
    best_model_path,
)

print(
    f"\nBest model based on F1 Score: "
    f"{best_model_name}"
)

print(
    f"Saved best model → "
    f"{best_model_path}"
)


# Save evaluation results

results_path = os.path.join(
    MODEL_DIR,
    "classification_results.csv",
)

results_df.to_csv(
    results_path,
    index=False,
)

print(
    f"Saved evaluation results → "
    f"{results_path}"
)


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("[6/6] FEATURE IMPORTANCE")
print("=" * 60)


# We use Random Forest for feature importance because
# tree-based models expose feature_importances_.

rf_pipeline = trained_models["random_forest"]

rf_model = rf_pipeline.named_steps["model"]

preprocessor = rf_pipeline.named_steps["preprocessor"]

feature_names = (
    preprocessor
    .get_feature_names_out()
)

importances = rf_model.feature_importances_

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importances,
    }
).sort_values(
    "Importance",
    ascending=False,
)

importance_path = os.path.join(
    MODEL_DIR,
    "feature_importance.csv",
)

importance_df.to_csv(
    importance_path,
    index=False,
)

print("\nTop Features:")

print(
    importance_df
    .head(15)
    .to_string(index=False)
)

print(
    f"\nSaved feature importance → "
    f"{importance_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"""
Dataset:
    {len(df):,} rows
    {X.shape[1]} input features

Models trained:
    Logistic Regression
    Random Forest
    XGBoost

Best model:
    {best_model_name}

Files generated:
    models/logistic_regression.joblib
    models/random_forest.joblib
    models/xgboost.joblib
    models/best_classifier.joblib
    models/classification_results.csv
    models/feature_importance.csv
""")

print("=" * 60)