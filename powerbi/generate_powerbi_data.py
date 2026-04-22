"""
Generates a clean Power BI summary CSV from processed data.
Run: python powerbi/generate_powerbi_data.py
"""
import pandas as pd, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "data", "processed_jobs.csv"))

# Clean up for Power BI
df["Fraud_Label"] = df["fraudulent"].map({0:"Real",1:"Fake"})
df["Work_Type"]   = df["telecommuting"].map({0:"On-site",1:"Work From Home"})
df["Company_Status"] = df["has_company"].map({0:"No Company",1:"Has Company"})
df["Salary_Status"]  = df["salary_anomaly"].map({0:"Normal Salary",1:"High Salary (Suspicious)"})

cols = [
    "job_id","title","company_profile","location","employment_type",
    "salary_range","max_salary","fraudulent","Fraud_Label",
    "trust_score","risk_level","missing_fields","scam_keyword_count",
    "has_company","has_company_logo","is_wfh","salary_anomaly",
    "description_length","Work_Type","Company_Status","Salary_Status"
]
out = df[cols]
out.to_csv(os.path.join(BASE, "powerbi", "FraudLens_PowerBI_Data.csv"), index=False)
print(f"✓ Power BI data saved: powerbi/FraudLens_PowerBI_Data.csv ({len(out)} rows)")
