"""
FraudLens – Dataset Generator
Combines a base fake-jobs style dataset with synthetic augmented data.
Run this first to create: fake_jobs_combined.csv
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# 1. BASE DATASET (mimics real Kaggle dataset)
# ─────────────────────────────────────────────
TITLES_REAL = [
    "Software Engineer", "Data Analyst", "Marketing Manager",
    "Product Manager", "HR Specialist", "Financial Analyst",
    "Customer Support Agent", "Sales Executive", "DevOps Engineer",
    "UI/UX Designer", "Content Writer", "Business Analyst",
    "Project Manager", "Operations Manager", "Graphic Designer",
]

TITLES_FAKE = [
    "Work From Home – Earn $5000/week", "Easy Money – No Experience Needed",
    "Unlimited Income Opportunity", "Online Data Entry – $80/hr",
    "Be Your Own Boss – Immediate Hire", "Make Money Fast – Part Time",
    "Secret Shopper – $500/day", "Bitcoin Trading Expert Needed",
    "Envelope Stuffing Job – High Pay", "Urgent – Wire Transfer Specialist",
]

COMPANIES_REAL = [
    "Infosys", "Wipro", "TCS", "HCL Technologies", "Cognizant",
    "Accenture", "IBM India", "Oracle India", "Microsoft India",
    "Amazon India", "Flipkart", "Paytm", "Zomato", "Swiggy", "HDFC Bank",
]

LOCATIONS = [
    "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Noida", "Gurgaon",
    "Remote", "Work From Home", "Anywhere",
]

DEPARTMENTS_REAL = [
    "Engineering", "Finance", "Marketing", "Operations",
    "Human Resources", "Sales", "Product", "Design", "IT",
]

EDUCATION_REAL = [
    "Bachelor's Degree", "Master's Degree", "B.Tech", "MBA",
    "B.Com", "BCA", "MCA", "PhD",
]

EMPLOYMENT_TYPES = ["Full-time", "Part-time", "Contract", "Temporary", "Other"]

REQUIRED_EXPERIENCE = [
    "Not Applicable", "Internship", "Entry level",
    "Mid-Senior level", "Director", "Executive",
]

REAL_DESCRIPTIONS = [
    "We are looking for a talented professional to join our growing team. "
    "You will work closely with cross-functional teams to deliver high-quality results. "
    "Strong communication skills and attention to detail are required. "
    "Experience with modern tools and frameworks is preferred.",

    "Join our dynamic team and contribute to meaningful projects. "
    "You will be responsible for analyzing business requirements and implementing solutions. "
    "We offer a collaborative work environment with opportunities for growth.",

    "As part of our team, you will design, develop, and maintain systems that power our platform. "
    "You should have a strong foundation in your field and be comfortable working in an agile environment.",

    "We are hiring an experienced professional to help us scale our operations. "
    "The ideal candidate will have a proven track record and excellent problem-solving skills.",
]

FAKE_DESCRIPTIONS = [
    "No experience needed! Earn up to $5000 per week working from home. "
    "Just sign up and start earning immediately. Limited spots available!",

    "Guaranteed income! Work only 2 hours per day and earn full-time salary. "
    "We pay weekly via PayPal or wire transfer. Apply now before positions are filled!",

    "Make real money from home. All you need is a phone and internet connection. "
    "Earn $500 daily. No boss, no office, no stress. Join thousands of happy workers!",

    "Urgent hiring! Wire transfer agents needed worldwide. "
    "Receive payments and transfer funds. High commission paid instantly. Confidential work.",
]

REAL_REQUIREMENTS = [
    "Bachelor's degree in relevant field. 2+ years of experience. "
    "Proficiency in MS Office. Strong analytical and communication skills.",

    "Minimum 3 years experience in the domain. Knowledge of industry tools. "
    "Ability to work in a team environment. Strong problem-solving skills.",

    "Relevant degree or equivalent experience. Excellent written and verbal communication. "
    "Ability to manage multiple tasks simultaneously.",
]

FAKE_REQUIREMENTS = [
    "No experience required. Just have a smartphone and internet.",
    "Must have PayPal or bank account to receive payments.",
    "Willing to work independently. No qualifications needed.",
    "",
]


def generate_real_job():
    has_company = random.random() > 0.05  # 95% have company
    salary_min = random.randint(30000, 120000)
    salary_max = salary_min + random.randint(10000, 50000)
    has_salary = random.random() > 0.3

    return {
        "job_id": None,
        "title": random.choice(TITLES_REAL),
        "location": random.choice(LOCATIONS[:10]),
        "department": random.choice(DEPARTMENTS_REAL),
        "salary_range": f"{salary_min}-{salary_max}" if has_salary else "",
        "company_profile": random.choice(COMPANIES_REAL) if has_company else "",
        "description": random.choice(REAL_DESCRIPTIONS),
        "requirements": random.choice(REAL_REQUIREMENTS),
        "benefits": random.choice([
            "Health insurance, PF, gratuity, flexible hours",
            "Annual bonus, remote work option, learning budget",
            "Medical cover, paid leave, team outings",
            "",
        ]),
        "telecommuting": random.choice([0, 1]),
        "has_company_logo": 1 if has_company and random.random() > 0.1 else 0,
        "has_questions": random.choice([0, 1]),
        "employment_type": random.choice(EMPLOYMENT_TYPES[:4]),
        "required_experience": random.choice(REQUIRED_EXPERIENCE[:5]),
        "required_education": random.choice(EDUCATION_REAL),
        "industry": random.choice(["Information Technology", "Finance", "Healthcare",
                                   "Education", "Manufacturing", "Retail", "BFSI"]),
        "function": random.choice(DEPARTMENTS_REAL),
        "fraudulent": 0,
    }


def generate_fake_job():
    # Fraud patterns: no company, crazy salary, vague desc, no requirements
    patterns = random.choice(["no_company", "high_salary", "vague_all", "urgent_wfh"])

    salary_min = random.randint(80000, 500000) if patterns == "high_salary" else random.randint(50000, 200000)
    salary_max = salary_min + random.randint(50000, 300000)
    has_salary = True  # fake jobs almost always show salary to lure

    return {
        "job_id": None,
        "title": random.choice(TITLES_FAKE),
        "location": random.choice(["Work From Home", "Anywhere", "Remote", "Online"]),
        "department": random.choice(["", "Other", "Sales"]),
        "salary_range": f"{salary_min}-{salary_max}" if has_salary else "",
        "company_profile": "" if patterns in ["no_company", "vague_all"] else "Anonymous",
        "description": random.choice(FAKE_DESCRIPTIONS),
        "requirements": random.choice(FAKE_REQUIREMENTS),
        "benefits": random.choice(["Unlimited earning potential", "Weekly payout", "", "Fast cash"]),
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0,
        "employment_type": random.choice(["Part-time", "Other", "Temporary"]),
        "required_experience": "Not Applicable",
        "required_education": random.choice(["", "Unspecified", "High School or equivalent"]),
        "industry": random.choice(["", "Other", "Online"]),
        "function": random.choice(["", "Other"]),
        "fraudulent": 1,
    }


def build_dataset(n_real=1500, n_fake=500):
    records = []
    for i in range(n_real):
        r = generate_real_job()
        r["job_id"] = i + 1
        records.append(r)
    for j in range(n_fake):
        r = generate_fake_job()
        r["job_id"] = n_real + j + 1
        records.append(r)

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["job_id"] = range(1, len(df) + 1)
    return df


if __name__ == "__main__":
    print("Generating dataset...")
    df = build_dataset(n_real=1500, n_fake=500)
    out = os.path.join(os.path.dirname(__file__), "fake_jobs_combined.csv")
    df.to_csv(out, index=False)
    print(f"Dataset saved: {out}")
    print(f"Total rows: {len(df)}")
    print(f"Fraudulent: {df['fraudulent'].sum()} | Real: {(df['fraudulent']==0).sum()}")
