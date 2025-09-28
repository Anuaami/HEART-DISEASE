# train_model.py
import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

warnings.filterwarnings("ignore")

DATA_CSV = "heart_failure_clinical_records_dataset.csv"
MODEL_OUT = "model.pkl"

if not Path(DATA_CSV).exists():
    raise FileNotFoundError(f"{DATA_CSV} not found. Put the CSV in the same folder as this script.")

# Load data
df = pd.read_csv(DATA_CSV)

if "DEATH_EVENT" not in df.columns:
    raise ValueError("Dataset must contain 'DEATH_EVENT' column as the target.")

X = df.drop(columns=["DEATH_EVENT"])
y = df["DEATH_EVENT"]

# Feature order (ensure this matches app input order)
FEATURE_ORDER = list(X.columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Try to import XGBClassifier; if not available, instruct user.
try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError(
        "XGBoost not installed. Install with `pip install xgboost` and retry."
    ) from e

# Tuned-ish params (balanced between speed and performance)
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", xgb)
])

print("Training XGBoost model (this may take some seconds)...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test ROC AUC : {roc:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Save model
with open(MODEL_OUT, "wb") as f:
    pickle.dump({
        "pipeline": pipeline,
        "feature_order": FEATURE_ORDER
    }, f)

print(f"\nSaved trained pipeline and feature_order to: {MODEL_OUT}")

