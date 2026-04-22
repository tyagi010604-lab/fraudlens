-- ============================================================
-- FraudLens – SQL Queries for Job Fraud Analysis
-- Database: SQLite (file: fraudlens.db) or MySQL compatible
-- Run via: python sql/load_to_db.py  (creates the DB first)
-- ============================================================

-- ─────────────────────────────────
-- TABLE SCHEMA (auto-created by Python script)
-- ─────────────────────────────────
/*
CREATE TABLE IF NOT EXISTS job_postings (
    job_id              INTEGER PRIMARY KEY,
    title               TEXT,
    location            TEXT,
    department          TEXT,
    salary_range        TEXT,
    company_profile     TEXT,
    description         TEXT,
    requirements        TEXT,
    benefits            TEXT,
    telecommuting       INTEGER,
    has_company_logo    INTEGER,
    has_questions       INTEGER,
    employment_type     TEXT,
    required_experience TEXT,
    required_education  TEXT,
    industry            TEXT,
    function            TEXT,
    fraudulent          INTEGER,
    -- Engineered features (added after preprocessing)
    description_length  INTEGER,
    has_company         INTEGER,
    missing_fields      INTEGER,
    salary_anomaly      INTEGER,
    trust_score         REAL,
    risk_level          TEXT
);
*/

-- ─────────────────────────────────
-- QUERY 1: Overall Fraud Distribution
-- ─────────────────────────────────
SELECT
    CASE WHEN fraudulent = 1 THEN 'Fake' ELSE 'Real' END AS job_type,
    COUNT(*) AS total,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM job_postings), 2) AS percentage
FROM job_postings
GROUP BY fraudulent;

-- ─────────────────────────────────
-- QUERY 2: Fraud by Employment Type
-- ─────────────────────────────────
SELECT
    employment_type,
    COUNT(*) AS total_jobs,
    SUM(fraudulent) AS fake_jobs,
    ROUND(SUM(fraudulent) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM job_postings
WHERE employment_type != ''
GROUP BY employment_type
ORDER BY fraud_rate_pct DESC;

-- ─────────────────────────────────
-- QUERY 3: Fraud by Telecommuting
-- ─────────────────────────────────
SELECT
    CASE WHEN telecommuting = 1 THEN 'Work From Home' ELSE 'On-site' END AS work_type,
    COUNT(*) AS total,
    SUM(fraudulent) AS fake,
    ROUND(SUM(fraudulent) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM job_postings
GROUP BY telecommuting;

-- ─────────────────────────────────
-- QUERY 4: Jobs Missing Company Profile (High Risk)
-- ─────────────────────────────────
SELECT
    CASE WHEN company_profile = '' OR company_profile IS NULL
         THEN 'No Company Info' ELSE 'Has Company Info' END AS company_status,
    COUNT(*) AS total,
    SUM(fraudulent) AS fake_count,
    ROUND(SUM(fraudulent) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM job_postings
GROUP BY company_status;

-- ─────────────────────────────────
-- QUERY 5: Salary Anomaly Analysis
-- ─────────────────────────────────
SELECT
    CASE
        WHEN salary_range = '' OR salary_range IS NULL THEN 'No Salary Listed'
        WHEN CAST(SUBSTR(salary_range, 1, INSTR(salary_range, '-') - 1) AS INTEGER) > 200000
             THEN 'Very High Salary (Suspicious)'
        ELSE 'Normal Salary Range'
    END AS salary_category,
    COUNT(*) AS total,
    SUM(fraudulent) AS fake_jobs,
    ROUND(SUM(fraudulent) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM job_postings
GROUP BY salary_category
ORDER BY fraud_rate_pct DESC;

-- ─────────────────────────────────
-- QUERY 6: Top Industries with Fraud
-- ─────────────────────────────────
SELECT
    COALESCE(NULLIF(industry, ''), 'Unknown') AS industry,
    COUNT(*) AS total,
    SUM(fraudulent) AS fake,
    ROUND(SUM(fraudulent) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM job_postings
GROUP BY industry
HAVING total > 5
ORDER BY fraud_rate_pct DESC
LIMIT 10;

-- ─────────────────────────────────
-- QUERY 7: Risk Level Distribution (after ML scoring)
-- ─────────────────────────────────
SELECT
    risk_level,
    COUNT(*) AS total,
    ROUND(AVG(trust_score), 1) AS avg_trust_score,
    SUM(fraudulent) AS actually_fake
FROM job_postings
WHERE risk_level IS NOT NULL
GROUP BY risk_level
ORDER BY avg_trust_score ASC;

-- ─────────────────────────────────
-- QUERY 8: Jobs with Most Missing Fields
-- ─────────────────────────────────
SELECT
    missing_fields,
    COUNT(*) AS total_jobs,
    SUM(fraudulent) AS fake_jobs,
    ROUND(SUM(fraudulent) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM job_postings
WHERE missing_fields IS NOT NULL
GROUP BY missing_fields
ORDER BY missing_fields DESC;

-- ─────────────────────────────────
-- QUERY 9: Has Company Logo vs Fraud
-- ─────────────────────────────────
SELECT
    CASE WHEN has_company_logo = 1 THEN 'Has Logo' ELSE 'No Logo' END AS logo_status,
    COUNT(*) AS total,
    SUM(fraudulent) AS fake,
    ROUND(SUM(fraudulent) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM job_postings
GROUP BY has_company_logo;

-- ─────────────────────────────────
-- QUERY 10: High-Risk Job Listings (for reporting)
-- ─────────────────────────────────
SELECT
    job_id,
    title,
    company_profile,
    location,
    salary_range,
    trust_score,
    risk_level
FROM job_postings
WHERE risk_level = 'High Risk'
ORDER BY trust_score ASC
LIMIT 20;

-- ─────────────────────────────────
-- QUERY 11: Monthly/Batch Summary View
-- ─────────────────────────────────
SELECT
    COUNT(*) AS total_jobs_analyzed,
    SUM(fraudulent) AS confirmed_fake,
    ROUND(AVG(trust_score), 1) AS avg_trust_score,
    SUM(CASE WHEN risk_level = 'High Risk' THEN 1 ELSE 0 END) AS high_risk_count,
    SUM(CASE WHEN risk_level = 'Medium Risk' THEN 1 ELSE 0 END) AS medium_risk_count,
    SUM(CASE WHEN risk_level = 'Low Risk' THEN 1 ELSE 0 END) AS low_risk_count
FROM job_postings;
