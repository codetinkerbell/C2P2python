# Baseline ML pipeline for Heart Disease prediction
# (Logistic Regression & Random Forest)

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import joblib
import sys

# -----------------------------
# 1️⃣ CSV file path (same folder as script)
# -----------------------------
script_dir = Path(__file__).parent
csv_path = script_dir / "CVD_cleaned(in).csv"

if not csv_path.exists():
    sys.exit(f"❌ CSV file not found: {csv_path}. Make sure it's in the same folder as this script.")

print(f"Loading data from: {csv_path}")

# -----------------------------
# 2️⃣ Load dataset
# -----------------------------
df = pd.read_csv(csv_path)
print("✅ CSV loaded successfully!")

# -----------------------------
# 3️⃣ Target column mapping
# -----------------------------
target_col = "Heart_Disease"
if target_col not in df.columns:
    sys.exit(f"❌ Column '{target_col}' not found in CSV! Check your header row.")

df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
print("\nAfter mapping, target value counts:")
print(df[target_col].value_counts())

# Class imbalance
n_pos = int(df[target_col].sum())
n_neg = int((df[target_col] == 0).sum())
print(f"\nPositive / Negative = {n_pos} / {n_neg} (pos fraction = {n_pos/len(df):.4f})")

# -----------------------------
# 4️⃣ Train/Test Split
# -----------------------------
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
print(f"\nTrain / Test sizes: {X_train.shape[0]} / {X_test.shape[0]}")

# -----------------------------
# 5️⃣ Features
# -----------------------------
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X_train.columns if c not in numeric_features]
print(f"Numeric features: {len(numeric_features)}, Categorical features: {len(categorical_features)}")

# -----------------------------
# 6️⃣ Preprocessing pipelines
# -----------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
], remainder="drop")

# -----------------------------
# 7️⃣ Models
# -----------------------------
log_pipe = Pipeline([
    ("preproc", preprocessor),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
])

rf_pipe = Pipeline([
    ("preproc", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
])

# -----------------------------
# 8️⃣ Cross-validation
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n🔹 Cross-validating Logistic Regression (ROC AUC)...")
lr_auc = cross_val_score(log_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"LR AUC mean/std: {lr_auc.mean():.4f} ± {lr_auc.std():.4f}")

print("\n🔹 Cross-validating Random Forest (ROC AUC)...")
rf_auc = cross_val_score(rf_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"RF AUC mean/std: {rf_auc.mean():.4f} ± {rf_auc.std():.4f}")

# -----------------------------
# 9️⃣ Train Random Forest on full training set
# -----------------------------
print("\n🚀 Training Random Forest on full training data...")
rf_pipe.fit(X_train, y_train)
y_prob = rf_pipe.predict_proba(X_test)[:, 1]
y_pred = rf_pipe.predict(X_test)

# -----------------------------
# 10️⃣ Evaluate
# -----------------------------
test_auc = roc_auc_score(y_test, y_prob)
print(f"\n✅ Test ROC AUC (Random Forest): {test_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC Curve
RocCurveDisplay.from_estimator(rf_pipe, X_test, y_test)
plt.title("ROC Curve - Random Forest (Test Set)")
plt.show()

# -----------------------------
# 11️⃣ Feature Importances
# -----------------------------
try:
    clf = rf_pipe.named_steps["clf"]
    num_cols = numeric_features
    ohe = rf_pipe.named_steps["preproc"].named_transformers_["cat"].named_steps["onehot"]
    cat_cols = list(ohe.get_feature_names_out(categorical_features))
    feat_names = num_cols + cat_cols

    importances = clf.feature_importances_
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(30)

    fi.plot(kind="barh", figsize=(8, 6))
    plt.gca().invert_yaxis()
    plt.title("Top 30 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"⚠ Feature importance extraction failed: {e}")

# -----------------------------
# 12️⃣ Save Model
# -----------------------------
model_path = script_dir / "best_model_HeartDisease_rf.pkl"
joblib.dump(rf_pipe, model_path)
print(f"\n💾 Model saved to: {model_path}")