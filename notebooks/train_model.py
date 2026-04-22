"""
FraudLens – Data Processing, EDA, Feature Engineering & ML Training
Run: python notebooks/train_model.py
Outputs:
  - models/fraud_model.pkl       (trained Random Forest)
  - models/feature_columns.pkl   (feature names for app)
  - data/processed_jobs.csv      (cleaned dataset with features)
  - outputs/eda_plots.png        (EDA charts)
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data", "fake_jobs_combined.csv")
OUT_D = os.path.join(BASE, "outputs")
MDL_D = os.path.join(BASE, "models")
os.makedirs(OUT_D, exist_ok=True)
os.makedirs(MDL_D, exist_ok=True)

# ─────────────────────────────────
# STEP 1 – LOAD & CLEAN
# ─────────────────────────────────
print("\n" + "="*55)
print("  FraudLens Model Training Pipeline")
print("="*55)

print("\n[1/6] Loading and cleaning data...")
df = pd.read_csv(DATA)
print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

# Fill nulls
text_cols = ["company_profile", "description", "requirements",
             "benefits", "salary_range", "department",
             "employment_type", "required_experience",
             "required_education", "industry", "function", "location"]
for c in text_cols:
    df[c] = df[c].fillna("").astype(str)

df["telecommuting"]     = df["telecommuting"].fillna(0).astype(int)
df["has_company_logo"]  = df["has_company_logo"].fillna(0).astype(int)
df["has_questions"]     = df["has_questions"].fillna(0).astype(int)
df["fraudulent"]        = df["fraudulent"].astype(int)

print(f"  Fraud: {df['fraudulent'].sum()} | Real: {(df['fraudulent']==0).sum()}")

# ─────────────────────────────────
# STEP 2 – FEATURE ENGINEERING
# ─────────────────────────────────
print("\n[2/6] Feature engineering...")

# 2a. Description length
df["description_length"] = df["description"].str.len()
df["requirements_length"] = df["requirements"].str.len()

# 2b. Company presence
df["has_company"] = (df["company_profile"].str.strip() != "").astype(int)

# 2c. Missing fields count (max 8 important fields)
def count_missing(row):
    fields = ["company_profile", "description", "requirements",
              "benefits", "salary_range", "department",
              "required_education", "industry"]
    return sum(1 for f in fields if str(row[f]).strip() == "")

df["missing_fields"] = df.apply(count_missing, axis=1)

# 2d. Salary anomaly – extract max salary from range string
def extract_max_salary(sal):
    try:
        parts = str(sal).split("-")
        nums = [int(p.strip().replace(",", "")) for p in parts if p.strip().replace(",","").isdigit()]
        return max(nums) if nums else 0
    except:
        return 0

df["max_salary"] = df["salary_range"].apply(extract_max_salary)
df["salary_anomaly"] = (df["max_salary"] > 200000).astype(int)
df["has_salary"] = (df["salary_range"].str.strip() != "").astype(int)

# 2e. Keyword flags (simple rule-based)
SCAM_KEYWORDS = [
    "earn", "guaranteed", "unlimited", "no experience",
    "work from home", "weekly pay", "easy money", "make money",
    "wire transfer", "paypal", "bitcoin", "urgent", "immediate"
]

def keyword_flag(text):
    t = str(text).lower()
    return sum(1 for kw in SCAM_KEYWORDS if kw in t)

df["scam_keyword_count"] = df["description"].apply(keyword_flag) + \
                           df["title"].apply(keyword_flag)

# 2f. Employment type encoding
emp_risk = {"Full-time": 0, "Contract": 0, "Part-time": 1,
            "Temporary": 1, "Other": 2, "": 1}
df["employment_risk"] = df["employment_type"].map(emp_risk).fillna(1).astype(int)

# 2g. WFH flag
df["is_wfh"] = df["telecommuting"].astype(int)

print("  Features created: description_length, has_company, missing_fields,")
print("                    salary_anomaly, scam_keyword_count, employment_risk, is_wfh")

# ─────────────────────────────────
# STEP 3 – TRUST SCORE (0-100)
# ─────────────────────────────────
print("\n[3/6] Building Trust Score...")

def compute_trust_score(row):
    score = 100  # start at full trust, deduct for red flags

    # Company missing → big red flag
    if row["has_company"] == 0:
        score -= 30
    # No company logo
    if row["has_company_logo"] == 0:
        score -= 10
    # Salary anomaly
    if row["salary_anomaly"] == 1:
        score -= 20
    # Many missing fields
    score -= row["missing_fields"] * 5
    # Scam keywords
    score -= row["scam_keyword_count"] * 8
    # Very short description (vague)
    if row["description_length"] < 100:
        score -= 15
    # WFH + no company = double suspicion
    if row["is_wfh"] == 1 and row["has_company"] == 0:
        score -= 10
    # High employment risk type
    score -= row["employment_risk"] * 5

    return max(0, min(100, score))

df["trust_score"] = df.apply(compute_trust_score, axis=1)

def classify_risk(ts):
    if ts >= 65:
        return "Low Risk"
    elif ts >= 35:
        return "Medium Risk"
    else:
        return "High Risk"

df["risk_level"] = df["trust_score"].apply(classify_risk)

print(f"  Avg Trust Score (Real): {df[df.fraudulent==0]['trust_score'].mean():.1f}")
print(f"  Avg Trust Score (Fake): {df[df.fraudulent==1]['trust_score'].mean():.1f}")

# ─────────────────────────────────
# STEP 4 – EDA PLOTS
# ─────────────────────────────────
print("\n[4/6] Generating EDA visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("FraudLens – Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.01)

palette = {"Real": "#2ecc71", "Fake": "#e74c3c"}
df["Label"] = df["fraudulent"].map({0: "Real", 1: "Fake"})

# Plot 1: Fraud distribution
ax1 = axes[0, 0]
counts = df["Label"].value_counts()
bars = ax1.bar(counts.index, counts.values,
               color=[palette[l] for l in counts.index], width=0.5, edgecolor="black")
for b in bars:
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+10,
             str(int(b.get_height())), ha="center", fontweight="bold")
ax1.set_title("Job Distribution: Real vs Fake", fontweight="bold")
ax1.set_ylabel("Count")

# Plot 2: Trust Score distribution
ax2 = axes[0, 1]
for label, color in palette.items():
    sub = df[df["Label"] == label]["trust_score"]
    ax2.hist(sub, bins=20, alpha=0.7, color=color, label=label, edgecolor="white")
ax2.set_title("Trust Score Distribution", fontweight="bold")
ax2.set_xlabel("Trust Score")
ax2.set_ylabel("Count")
ax2.legend()

# Plot 3: Company presence vs fraud
ax3 = axes[0, 2]
cp_fraud = df.groupby(["has_company", "Label"]).size().unstack(fill_value=0)
cp_fraud.index = ["No Company", "Has Company"]
cp_fraud.plot(kind="bar", ax=ax3, color=[palette["Fake"], palette["Real"]],
              edgecolor="black", rot=0)
ax3.set_title("Company Presence vs Fraud", fontweight="bold")
ax3.set_ylabel("Count")

# Plot 4: Scam keyword count
ax4 = axes[1, 0]
for label, color in palette.items():
    sub = df[df["Label"] == label]["scam_keyword_count"]
    ax4.hist(sub, bins=15, alpha=0.7, color=color, label=label, edgecolor="white")
ax4.set_title("Scam Keyword Count", fontweight="bold")
ax4.set_xlabel("Count of Suspicious Keywords")
ax4.legend()

# Plot 5: Missing fields vs fraud rate
ax5 = axes[1, 1]
mf = df.groupby("missing_fields")["fraudulent"].mean().reset_index()
ax5.bar(mf["missing_fields"], mf["fraudulent"]*100,
        color="#e74c3c", edgecolor="black", alpha=0.8)
ax5.set_title("Missing Fields → Fraud Rate", fontweight="bold")
ax5.set_xlabel("Number of Missing Fields")
ax5.set_ylabel("Fraud Rate (%)")

# Plot 6: Risk level breakdown
ax6 = axes[1, 2]
risk_counts = df["risk_level"].value_counts()
colors_risk = {"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}
ax6.pie(risk_counts.values, labels=risk_counts.index,
        colors=[colors_risk[r] for r in risk_counts.index],
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2})
ax6.set_title("Risk Level Distribution", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT_D, "eda_plots.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/eda_plots.png")

# ─────────────────────────────────
# STEP 5 – MACHINE LEARNING
# ─────────────────────────────────
print("\n[5/6] Training Machine Learning model...")

FEATURE_COLS = [
    "description_length", "requirements_length", "has_company",
    "has_company_logo", "has_questions", "missing_fields",
    "salary_anomaly", "has_salary", "scam_keyword_count",
    "employment_risk", "is_wfh", "max_salary",
]

X = df[FEATURE_COLS]
y = df["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest (primary model)
rf = RandomForestClassifier(n_estimators=150, max_depth=10,
                             random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("\n  ── Model Evaluation (Random Forest) ──")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
auc = roc_auc_score(y_test, y_proba)
print(f"  ROC-AUC Score: {auc:.4f}")

# Confusion matrix plot
fig2, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
            linewidths=1, ax=ax)
ax.set_title("Confusion Matrix – FraudLens RF Model", fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT_D, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()

# Feature importance plot
fig3, ax = plt.subplots(figsize=(8, 5))
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
importances.plot(kind="barh", ax=ax, color="#2c3e50", edgecolor="white")
ax.set_title("Feature Importance – Random Forest", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(OUT_D, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/confusion_matrix.png, outputs/feature_importance.png")

# ─────────────────────────────────
# STEP 6 – SAVE MODEL & DATA
# ─────────────────────────────────
print("\n[6/6] Saving model and processed dataset...")

with open(os.path.join(MDL_D, "fraud_model.pkl"), "wb") as f:
    pickle.dump(rf, f)

with open(os.path.join(MDL_D, "feature_columns.pkl"), "wb") as f:
    pickle.dump(FEATURE_COLS, f)

# Save processed CSV (also update SQLite)
processed_path = os.path.join(BASE, "data", "processed_jobs.csv")
df.to_csv(processed_path, index=False)

# Update SQLite with engineered columns
import sqlite3
DB = os.path.join(BASE, "data", "fraudlens.db")
conn = sqlite3.connect(DB)
df.to_sql("job_postings", conn, if_exists="replace", index=False)
conn.commit(); conn.close()

print("  Saved: models/fraud_model.pkl")
print("  Saved: models/feature_columns.pkl")
print("  Saved: data/processed_jobs.csv")
print("  Updated: data/fraudlens.db")
print("\n✓ Training complete!")
