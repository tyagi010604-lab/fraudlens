"""
FraudLens – SQL Database Loader
Loads the CSV dataset into SQLite and runs all analysis queries.
Run: python sql/load_to_db.py
"""

import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = os.path.join(BASE_DIR, "data", "fake_jobs_combined.csv")
DB_PATH   = os.path.join(BASE_DIR, "data", "fraudlens.db")
SQL_PATH  = os.path.join(BASE_DIR, "sql",  "fraud_queries.sql")


def load_data_to_db():
    print("Loading CSV into SQLite database...")
    df = pd.read_csv(CSV_PATH)

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("job_postings", conn, if_exists="replace", index=False)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM job_postings").fetchone()[0]
    print(f"  ✓ {count} records loaded into 'job_postings' table")
    return conn


def run_analysis_queries(conn):
    queries = {
        "Fraud Distribution": """
            SELECT
                CASE WHEN fraudulent=1 THEN 'Fake' ELSE 'Real' END AS job_type,
                COUNT(*) AS total,
                ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM job_postings),2) AS pct
            FROM job_postings GROUP BY fraudulent
        """,
        "Fraud by Employment Type": """
            SELECT employment_type,
                   COUNT(*) AS total,
                   SUM(fraudulent) AS fake,
                   ROUND(SUM(fraudulent)*100.0/COUNT(*),2) AS fraud_pct
            FROM job_postings
            WHERE employment_type != ''
            GROUP BY employment_type ORDER BY fraud_pct DESC
        """,
        "Telecommuting vs Fraud": """
            SELECT CASE WHEN telecommuting=1 THEN 'WFH' ELSE 'On-site' END AS work_type,
                   COUNT(*) AS total, SUM(fraudulent) AS fake,
                   ROUND(SUM(fraudulent)*100.0/COUNT(*),2) AS fraud_pct
            FROM job_postings GROUP BY telecommuting
        """,
        "Company Missing = Fraud?": """
            SELECT CASE WHEN company_profile='' OR company_profile IS NULL
                        THEN 'No Company' ELSE 'Has Company' END AS co_status,
                   COUNT(*) AS total,
                   SUM(fraudulent) AS fake,
                   ROUND(SUM(fraudulent)*100.0/COUNT(*),2) AS fraud_pct
            FROM job_postings GROUP BY co_status
        """,
        "Logo Presence vs Fraud": """
            SELECT CASE WHEN has_company_logo=1 THEN 'Has Logo' ELSE 'No Logo' END AS logo,
                   COUNT(*) AS total, SUM(fraudulent) AS fake,
                   ROUND(SUM(fraudulent)*100.0/COUNT(*),2) AS fraud_pct
            FROM job_postings GROUP BY has_company_logo
        """,
    }

    print("\n" + "="*60)
    print("  SQL ANALYSIS RESULTS")
    print("="*60)
    for title, q in queries.items():
        print(f"\n── {title} ──")
        df = pd.read_sql_query(q, conn)
        print(df.to_string(index=False))

    conn.close()
    print("\n✓ All queries complete. Database saved to:", DB_PATH)


if __name__ == "__main__":
    conn = load_data_to_db()
    run_analysis_queries(conn)
