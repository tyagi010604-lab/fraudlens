# 🔍 FraudLens – Fake Job Detection & Job Trust Analytics System

A complete, end-to-end data analytics and machine learning project demonstrating
real-world data analyst skills.

---

## 📁 Project Structure

```
FraudLens/
├── data/
│   ├── generate_data.py          ← Step 1: Creates the dataset
│   ├── fake_jobs_combined.csv    ← Raw combined dataset (2000 rows)
│   ├── processed_jobs.csv        ← Cleaned dataset with features + scores
│   └── fraudlens.db              ← SQLite database
│
├── sql/
│   ├── load_to_db.py             ← Loads CSV into SQLite + runs queries
│   └── fraud_queries.sql         ← All SQL analysis queries
│
├── notebooks/
│   ├── train_model.py            ← EDA + Feature Engineering + ML Training
│   └── generate_excel.py         ← Creates Excel workbook with pivot tables
│
├── models/
│   ├── fraud_model.pkl           ← Trained Random Forest model
│   └── feature_columns.pkl       ← Feature names (used by app)
│
├── app/
│   └── app.py                    ← Streamlit web application (FraudLens UI)
│
├── powerbi/
│   ├── FraudLens_PowerBI_Data.csv ← Clean data for Power BI
│   ├── generate_powerbi_data.py  ← Script to regenerate Power BI data
│   └── PowerBI_Setup_Guide.md    ← Step-by-step Power BI dashboard guide
│
├── outputs/
│   ├── eda_plots.png             ← EDA visualizations (6 charts)
│   ├── confusion_matrix.png      ← Model confusion matrix
│   ├── feature_importance.png    ← RF feature importance chart
│   └── FraudLens_Analysis.xlsx  ← Excel workbook (pivot tables + charts)
│
├── requirements.txt              ← Python dependencies
└── README.md                     ← This file
```

---

## 🚀 Quick Start (Run in Order)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python data/generate_data.py
```

### 3. Load data into SQLite + run SQL analysis
```bash
python sql/load_to_db.py
```

### 4. Train ML model + generate EDA charts
```bash
python notebooks/train_model.py
```

### 5. Generate Excel report
```bash
python notebooks/generate_excel.py
```

### 6. Launch the web app
```bash
streamlit run app/app.py
```

### 7. Power BI Dashboard
- Open Power BI Desktop
- Import `powerbi/FraudLens_PowerBI_Data.csv`
- Follow `powerbi/PowerBI_Setup_Guide.md`

---

## 🧠 What Each Component Does

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Dataset | Python (Pandas) | 2000 synthetic job postings (real + fake) |
| SQL Analysis | SQLite + Pandas | Fraud trend queries |
| Feature Engineering | Pandas + NumPy | 12 fraud signals extracted |
| EDA | Matplotlib + Seaborn | 6 visual insights |
| Trust Score | Rule-based (Python) | 0–100 score per job |
| ML Model | Random Forest (sklearn) | Predicts fake vs real |
| Excel Report | OpenPyXL | Pivot tables + charts |
| Web App | Streamlit | Interactive fraud detector |
| Power BI | Power BI Desktop | Executive dashboard |

---

## 📊 Model Performance

- Algorithm: **Random Forest** (150 trees, max depth 10)
- Test Accuracy: **100%**
- ROC-AUC Score: **1.00**
- Features used: **12 engineered signals**

---

## 🔍 Key Fraud Signals

1. **No company name** → 80% fraud rate
2. **No company logo** → 70% fraud rate  
3. **Salary > ₹2L/month** → suspicious
4. **Scam keywords** (earn, guaranteed, no experience...) → high risk
5. **WFH + no company** → very high risk
6. **Many missing fields** → suspicious
7. **Very short description** → vague = suspicious

---

## 🏆 Resume Skills Demonstrated

- Data Collection & Synthetic Data Generation
- SQL Database Design & Query Writing
- Data Cleaning & Feature Engineering (Pandas, NumPy)
- Exploratory Data Analysis & Visualization
- Machine Learning (Classification, Evaluation)
- Business Intelligence (Excel, Power BI)
- Web Application Development (Streamlit)
- End-to-End Project Delivery

---

## Author
Built as a portfolio-ready data analytics project.
