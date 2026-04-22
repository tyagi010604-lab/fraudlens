"""
FraudLens – Streamlit Web Application (v2.0)
Run: streamlit run app/app.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDL_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

@st.cache_resource
def load_model():
    with open(os.path.join(MDL_DIR, "fraud_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MDL_DIR, "feature_columns.pkl"), "rb") as f:
        features = pickle.load(f)
    return model, features

@st.cache_data
def load_dataset():
    return pd.read_csv(os.path.join(DATA_DIR, "processed_jobs.csv"))

model, FEATURE_COLS = load_model()
df_all = load_dataset()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="FraudLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2a3a;
}
section[data-testid="stSidebar"] * { color: #c8d6e5 !important; }

.sidebar-stat-box {
    background: #1e2a3a;
    border: 1px solid #2c3e55;
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 0.82rem;
    color: #8da9c4 !important;
    line-height: 2.0;
}
.sidebar-stat-box b { color: #5dade2 !important; }
.sidebar-stat-box .stat-title {
    color: #5dade2 !important;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    display: block;
}

/* ── Header ── */
.fl-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2942 100%);
    border-bottom: 1px solid #1e3a5f;
    padding: 22px 30px 18px;
    margin: -1rem -1rem 1.5rem -1rem;
    border-radius: 0 0 8px 8px;
}
.fl-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1.5px;
}
.fl-logo span { color: #5dade2; }
.fl-tagline { color: #6e8fa8; font-size: 0.9rem; margin-top: 2px; }
.fl-badge {
    display: inline-block;
    background: #1a3a5c;
    color: #5dade2;
    border: 1px solid #2e5f8a;
    border-radius: 20px;
    padding: 2px 11px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 10px;
    vertical-align: middle;
}

/* ── Section titles ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e8f0f7;
    margin-bottom: 2px;
}
.section-sub { color: #6e8fa8; font-size: 0.88rem; margin-bottom: 16px; }

/* ── Result cards ── */
.result-card {
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
    border: 1.5px solid;
}
.result-high   { background: #1f0d0d; border-color: #c0392b; }
.result-medium { background: #1f1700; border-color: #d68910; }
.result-low    { background: #0d1f12; border-color: #1e8449; }
.result-icon   { font-size: 2.6rem; margin-bottom: 4px; }
.result-label  { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #eaf0f6; }
.result-sub    { color: #7f9ab0; font-size: 0.82rem; margin-top: 4px; }

/* ── Trust box ── */
.trust-box {
    background: #13202f;
    border: 1.5px solid #1e3a5f;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.trust-number { font-family: 'Syne', sans-serif; font-size: 3.6rem; font-weight: 800; line-height: 1; }
.trust-label  { color: #7f9ab0; font-size: 0.85rem; margin-top: 6px; }

/* ── Reason badges ── */
.badge-wrap { margin-top: 8px; }
.rbadge {
    display: inline-block;
    padding: 5px 13px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 4px 4px 4px 0;
    letter-spacing: 0.3px;
}
.rbadge-red    { background: #3d1212; color: #e74c3c; border: 1px solid #c0392b; }
.rbadge-orange { background: #2e1c00; color: #e67e22; border: 1px solid #d35400; }
.rbadge-green  { background: #0d2d18; color: #2ecc71; border: 1px solid #1e8449; }

/* ── Buttons ── */
.stButton > button {
    background: #1a4a7a !important;
    color: #eaf0f6 !important;
    border: 1px solid #2980b9 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    padding: 10px !important;
}
.stButton > button:hover { background: #2471a3 !important; }

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: #13202f;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 12px 16px;
}
div[data-testid="stMetric"] label { color: #6e8fa8 !important; font-size: 0.8rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #eaf0f6 !important; }

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background: #13202f !important;
    color: #eaf0f6 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}
label[data-testid="stWidgetLabel"] { color: #8da9c4 !important; font-size: 0.88rem !important; }
.stCheckbox label { color: #c8d6e5 !important; }
hr { border-color: #1e2a3a !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="fl-header">
  <div class="fl-logo">Fraud<span>Lens</span>
  </div>
  <div class="fl-tagline">Fake Job Detection &amp; Trust Analytics </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigation",
    ["🔍 Detect Fraud", "📊 Analytics Dashboard",
     "⚖️ Compare Two Jobs", "📦 Batch Analyzer"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="sidebar-stat-box">
  <span class="stat-title">⚡ FraudLens Stats</span>
  <b>Dataset</b>&nbsp;&nbsp; · 2,000 job postings<br>
  <b>Model</b>&nbsp;&nbsp;&nbsp;&nbsp; · Random Forest<br>
  <b>AUC</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; · 1.00<br>
  <b>Features</b>&nbsp; · 12 engineered signals<br>
  <b>Fraud Rate</b> · 25% in dataset
</div>
""", unsafe_allow_html=True)

# ── Shared helpers ─────────────────────────────────────────────
SCAM_KEYWORDS = [
    "earn","guaranteed","unlimited","no experience","work from home",
    "weekly pay","easy money","make money","wire transfer","paypal",
    "bitcoin","urgent","immediate","no qualification","daily payment","free training"
]

def kw_flag(text):
    t = str(text).lower()
    return sum(1 for kw in SCAM_KEYWORDS if kw in t)

def extract_max_salary(s):
    try:
        parts = str(s).split("-")
        nums = [int(p.strip().replace(",","")) for p in parts if p.strip().replace(",","").isdigit()]
        return max(nums) if nums else 0
    except:
        return 0

def build_features(title, company, description, requirements,
                   salary_range, employment, telecommuting, has_logo, has_questions):
    desc_len    = len(description)
    req_len     = len(requirements)
    has_company = 1 if company.strip() else 0
    max_salary  = extract_max_salary(salary_range)
    salary_anom = 1 if max_salary > 200000 else 0
    has_sal     = 1 if salary_range.strip() else 0
    kw_count    = kw_flag(description) + kw_flag(title)
    emp_risk_map = {"Full-time":0,"Contract":0,"Part-time":1,"Temporary":1,"Other":2}
    emp_risk    = emp_risk_map.get(employment, 1)
    is_wfh      = 1 if telecommuting else 0
    missing     = sum(1 for v in [company, description, requirements, salary_range]
                      if not str(v).strip())
    return pd.DataFrame([[
        desc_len, req_len, has_company,
        1 if has_logo else 0, 1 if has_questions else 0,
        missing, salary_anom, has_sal, kw_count, emp_risk, is_wfh, max_salary
    ]], columns=FEATURE_COLS), {
        "desc_len":desc_len,"has_company":has_company,
        "max_salary":max_salary,"salary_anom":salary_anom,
        "kw_count":kw_count,"missing":missing,"is_wfh":is_wfh,
        "employment":employment,"has_logo":has_logo,
        "has_questions":has_questions,"req_len":req_len
    }

def compute_trust(vals):
    score = 100
    if vals["has_company"] == 0:                        score -= 30
    if not vals["has_logo"]:                            score -= 10
    if vals["salary_anom"]:                             score -= 20
    score -= vals["missing"] * 5
    score -= vals["kw_count"] * 8
    if vals["desc_len"] < 100:                          score -= 15
    if vals["is_wfh"] and vals["has_company"] == 0:    score -= 10
    if vals["employment"] in ["Other","Temporary"]:     score -= 5
    return max(0, min(100, score))

def get_reasons(vals):
    r = []
    if vals["has_company"] == 0:                        r.append(("🏢 No company name",              "red"))
    if not vals["has_logo"]:                            r.append(("🖼️ No company logo",              "orange"))
    if vals["salary_anom"]:                             r.append((f"💰 High salary ₹{vals['max_salary']:,}", "red"))
    if vals["kw_count"] > 0:                            r.append((f"🚨 {vals['kw_count']} scam keyword(s)", "red"))
    if vals["desc_len"] < 100:                          r.append(("📝 Very short description",        "orange"))
    if vals["is_wfh"] and vals["has_company"] == 0:    r.append(("🏠 WFH + no company info",         "red"))
    if vals["missing"] >= 3:                            r.append((f"❓ {vals['missing']} fields missing", "orange"))
    if vals["employment"] in ["Other","Temporary"]:     r.append((f"📋 Risk type: {vals['employment']}", "orange"))
    if vals["req_len"] < 30:                            r.append(("📋 No requirements listed",        "orange"))
    if not r:                                           r.append(("✅ No major red flags",             "green"))
    return r

def make_gauge(trust_score):
    ts_color = ("#1e8449" if trust_score>=65 else "#d68910" if trust_score>=35 else "#c0392b")
    fig, ax = plt.subplots(figsize=(7, 1.1))
    fig.patch.set_facecolor("#13202f")
    ax.set_facecolor("#13202f")
    ax.barh([""], [100], color="#1a2a3a", height=0.55, edgecolor="none")
    ax.barh([""], [trust_score], color=ts_color, height=0.55, edgecolor="none")
    ax.axvline(35, color="#c0392b", linestyle="--", alpha=0.4, lw=1.2)
    ax.axvline(65, color="#1e8449", linestyle="--", alpha=0.4, lw=1.2)
    ax.text(17, 0.38, "High Risk", ha="center", color="#c0392b", fontsize=7)
    ax.text(50, 0.38, "Medium",    ha="center", color="#d68910", fontsize=7)
    ax.text(82, 0.38, "Low Risk",  ha="center", color="#1e8449", fontsize=7)
    ax.text(trust_score, -0.38, f"▲ {trust_score}", ha="center",
            color=ts_color, fontsize=9, fontweight="bold")
    ax.set_xlim(0, 100); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors="#6e8fa8", labelsize=7)
    plt.tight_layout(pad=0.2)
    return fig

def render_result(prob_fake, trust_score, reasons):
    risk      = ("High Risk" if trust_score < 35 else "Medium Risk" if trust_score < 65 else "Low Risk")
    icon      = {"High Risk":"🚨","Medium Risk":"⚠️","Low Risk":"✅"}[risk]
    card_cls  = {"High Risk":"result-high","Medium Risk":"result-medium","Low Risk":"result-low"}[risk]
    ts_color  = ("#1e8449" if trust_score>=65 else "#d68910" if trust_score>=35 else "#c0392b")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="result-card {card_cls}">
          <div class="result-icon">{icon}</div>
          <div class="result-label">{risk}</div>
          <div class="result-sub">Risk Classification</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.metric("🎯 Fake Probability", f"{prob_fake*100:.1f}%")
        st.metric("✅ Real Probability", f"{(1-prob_fake)*100:.1f}%")
    with r3:
        st.markdown(f"""
        <div class="trust-box">
          <div class="trust-number" style="color:{ts_color}">{trust_score}</div>
          <div class="trust-label">Trust Score / 100<br>
          <span style="font-size:0.75rem">Higher = More Trustworthy</span></div>
        </div>""", unsafe_allow_html=True)

    st.pyplot(make_gauge(trust_score), use_container_width=True)
    plt.close()

    st.markdown("**🚩 Reasons for Assessment:**")
    badge_map = {"red":"rbadge-red","orange":"rbadge-orange","green":"rbadge-green"}
    badges = " ".join([f'<span class="rbadge {badge_map[c]}">{txt}</span>' for txt, c in reasons])
    st.markdown(f'<div class="badge-wrap">{badges}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# PAGE 1 – DETECT FRAUD
# ══════════════════════════════════════════════════
if page == "🔍 Detect Fraud":
    st.markdown('<div class="section-title">🔍 Analyze a Job Posting</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Paste details from any job listing to check for fraud signals</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        title         = st.text_input("Job Title ✱",       placeholder="e.g. Software Engineer")
        company       = st.text_input("Company Name",       placeholder="Leave blank if not listed")
        location      = st.text_input("Location",           placeholder="e.g. Bengaluru / Remote")
        employment    = st.selectbox("Employment Type",     ["Full-time","Part-time","Contract","Temporary","Other"])
        telecommuting = st.checkbox("🏠 Work From Home / Remote?")
    with c2:
        description   = st.text_area("Job Description ✱",  placeholder="Paste the job description here...", height=140)
        requirements  = st.text_area("Job Requirements",    placeholder="Qualifications, experience, skills...", height=80)
        salary_range  = st.text_input("Salary Range",       placeholder="e.g. 60000-90000")
        has_logo      = st.checkbox("🖼️ Listing has a company logo?")
        has_questions = st.checkbox("❓ Listing includes screening questions?")

    st.markdown("")
    if st.button("🔍 Analyze Job Posting"):
        if not title or not description:
            st.error("Please fill in at least Job Title and Description.")
        else:
            feats, vals = build_features(title, company, description, requirements,
                                         salary_range, employment, telecommuting, has_logo, has_questions)
            prob_fake   = model.predict_proba(feats)[0][1]
            trust_score = compute_trust(vals)
            reasons     = get_reasons(vals)
            st.markdown("---")
            st.markdown('<div class="section-title">📋 Analysis Results</div>', unsafe_allow_html=True)
            render_result(prob_fake, trust_score, reasons)
            st.markdown("---")
            st.caption("ℹ️ FraudLens uses ML + rule-based scoring. Always verify through official company channels.")


# ══════════════════════════════════════════════════
# PAGE 2 – ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.markdown('<div class="section-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Insights from {len(df_all):,} analyzed job postings</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Jobs",       f"{len(df_all):,}")
    k2.metric("Fake Jobs",        f"{df_all['fraudulent'].sum():,}")
    k3.metric("Fraud Rate",       f"{df_all['fraudulent'].mean()*100:.0f}%")
    k4.metric("Avg Trust (Real)", f"{df_all[df_all.fraudulent==0]['trust_score'].mean():.0f}/100")
    k5.metric("High Risk Jobs",   f"{(df_all.risk_level=='High Risk').sum():,}")

    st.markdown("---")
    DARK_BG, PLOT_BG, TEXT_C, GRID_C = "#0f1117", "#13202f", "#c8d6e5", "#1e2a3a"
    R, F = "#2ecc71", "#e74c3c"
    df_all["Label"] = df_all["fraudulent"].map({0:"Real",1:"Fake"})

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes.flat:
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=TEXT_C, labelsize=8)
        for sp in ax.spines.values(): sp.set_color(GRID_C)

    # 1. Real vs Fake
    ax = axes[0,0]
    cnts = df_all["fraudulent"].value_counts()
    bars = ax.bar(["Real","Fake"],[cnts.get(0,0),cnts.get(1,0)],color=[R,F],width=0.45,edgecolor=DARK_BG)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+15, str(int(b.get_height())),
                ha="center", color=TEXT_C, fontsize=9, fontweight="bold")
    ax.set_title("Real vs Fake Jobs", color=TEXT_C, fontweight="bold", pad=10)
    ax.set_ylabel("Count", color=TEXT_C)
    ax.yaxis.grid(True, color=GRID_C, linestyle="--", alpha=0.5); ax.set_axisbelow(True)

    # 2. Trust score histogram
    ax = axes[0,1]
    ax.hist(df_all[df_all.fraudulent==0]["trust_score"],bins=20,color=R,alpha=0.8,label="Real",edgecolor=DARK_BG)
    ax.hist(df_all[df_all.fraudulent==1]["trust_score"],bins=20,color=F,alpha=0.8,label="Fake",edgecolor=DARK_BG)
    ax.set_title("Trust Score Distribution", color=TEXT_C, fontweight="bold", pad=10)
    ax.set_xlabel("Trust Score", color=TEXT_C)
    ax.legend(facecolor=PLOT_BG, edgecolor=GRID_C, labelcolor=TEXT_C)

    # 3. Company presence
    ax = axes[0,2]
    cp = df_all.groupby(["has_company","fraudulent"]).size().unstack(fill_value=0)
    x = np.arange(2)
    r0 = [cp[0].get(0,0), cp[0].get(1,0)]
    r1 = [cp[1].get(0,0), cp[1].get(1,0)]
    ax.bar(x-0.2, r0, 0.38, label="Real", color=R, edgecolor=DARK_BG)
    ax.bar(x+0.2, r1, 0.38, label="Fake", color=F, edgecolor=DARK_BG)
    ax.set_xticks(x); ax.set_xticklabels(["No Company","Has Company"], color=TEXT_C)
    ax.set_title("Company Presence vs Fraud", color=TEXT_C, fontweight="bold", pad=10)
    ax.legend(facecolor=PLOT_BG, edgecolor=GRID_C, labelcolor=TEXT_C)
    ax.yaxis.grid(True, color=GRID_C, linestyle="--", alpha=0.5); ax.set_axisbelow(True)

    # 4. Employment type fraud rate
    ax = axes[1,0]
    et = df_all[df_all.employment_type!=""].groupby("employment_type")["fraudulent"].mean()*100
    et = et.sort_values()
    ce = [F if v>30 else "#f39c12" if v>10 else R for v in et.values]
    ax.barh(et.index, et.values, color=ce, edgecolor=DARK_BG)
    ax.set_title("Fraud Rate by Employment Type", color=TEXT_C, fontweight="bold", pad=10)
    ax.set_xlabel("Fraud Rate %", color=TEXT_C)
    ax.tick_params(axis='y', colors=TEXT_C)
    ax.xaxis.grid(True, color=GRID_C, linestyle="--", alpha=0.5); ax.set_axisbelow(True)

    # 5. Missing fields fraud rate
    ax = axes[1,1]
    mf = df_all.groupby("missing_fields")["fraudulent"].mean().reset_index()
    bc = [F if v>0.4 else "#f39c12" if v>0.2 else R for v in mf["fraudulent"]]
    ax.bar(mf["missing_fields"], mf["fraudulent"]*100, color=bc, edgecolor=DARK_BG)
    ax.set_title("Missing Fields → Fraud Rate", color=TEXT_C, fontweight="bold", pad=10)
    ax.set_xlabel("# Missing Fields", color=TEXT_C); ax.set_ylabel("Fraud Rate %", color=TEXT_C)
    ax.yaxis.grid(True, color=GRID_C, linestyle="--", alpha=0.5); ax.set_axisbelow(True)

    # 6. Risk pie
    ax = axes[1,2]
    rc = df_all["risk_level"].value_counts()
    rc_c = [{"Low Risk":R,"Medium Risk":"#f39c12","High Risk":F}.get(r,"#7f8c8d") for r in rc.index]
    wedges, texts, autotexts = ax.pie(rc.values, labels=rc.index, colors=rc_c,
                                       autopct="%1.1f%%", startangle=90,
                                       wedgeprops={"edgecolor":DARK_BG,"linewidth":2})
    for t in texts: t.set_color(TEXT_C)
    for t in autotexts: t.set_color("#0f1117"); t.set_fontweight("bold")
    ax.set_title("Risk Level Distribution", color=TEXT_C, fontweight="bold", pad=10)

    plt.tight_layout(pad=2.0)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.markdown("**📋 High Risk Job Listings (Top 20)**")
    hr = df_all[df_all.risk_level=="High Risk"][
        ["title","company_profile","location","trust_score","risk_level","scam_keyword_count","fraudulent"]
    ].head(20).rename(columns={"company_profile":"Company","scam_keyword_count":"Scam KWs",
                                "trust_score":"Trust","fraudulent":"Is Fake"})
    st.dataframe(hr, use_container_width=True)


# ══════════════════════════════════════════════════
# PAGE 3 – COMPARE TWO JOBS
# ══════════════════════════════════════════════════
elif page == "⚖️ Compare Two Jobs":
    st.markdown('<div class="section-title">⚖️ Side-by-Side Job Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Paste two job listings and instantly compare their fraud signals and trust scores</div>', unsafe_allow_html=True)

    col_a, spacer, col_b = st.columns([5, 0.3, 5])

    def job_form(col, label, key):
        with col:
            st.markdown(f"**{label}**")
            t   = st.text_input("Job Title",       key=f"title_{key}",   placeholder="e.g. Data Analyst")
            co  = st.text_input("Company",          key=f"company_{key}", placeholder="Leave blank if missing")
            d   = st.text_area("Description",       key=f"desc_{key}",    placeholder="Paste description...", height=120)
            r   = st.text_area("Requirements",      key=f"req_{key}",     placeholder="Qualifications...", height=60)
            s   = st.text_input("Salary Range",     key=f"sal_{key}",     placeholder="e.g. 40000-80000")
            e   = st.selectbox("Employment Type",   ["Full-time","Part-time","Contract","Temporary","Other"], key=f"emp_{key}")
            wfh = st.checkbox("Work From Home?",    key=f"wfh_{key}")
            lg  = st.checkbox("Has Company Logo?",  key=f"logo_{key}")
            q   = st.checkbox("Has Questions?",     key=f"q_{key}")
        return t, co, d, r, s, e, wfh, lg, q

    inputs_a = job_form(col_a, "🔵 Job A", "a")
    with spacer:
        st.markdown("<br>"*12 + "<div style='text-align:center;color:#1e3a5f;font-size:2rem'>⚡</div>",
                    unsafe_allow_html=True)
    inputs_b = job_form(col_b, "🟠 Job B", "b")

    st.markdown("")
    if st.button("⚖️ Compare Both Jobs"):
        results = []
        for inputs, lbl in [(inputs_a,"Job A"),(inputs_b,"Job B")]:
            t,co,d,r,s,e,wfh,lg,q = inputs
            if not t or not d:
                st.warning(f"Please fill in Title and Description for {lbl}.")
                st.stop()
            feats, vals = build_features(t,co,d,r,s,e,wfh,lg,q)
            pf  = model.predict_proba(feats)[0][1]
            ts  = compute_trust(vals)
            rsn = get_reasons(vals)
            results.append((lbl, pf, ts, rsn))

        st.markdown("---")
        st.markdown('<div class="section-title">📊 Comparison Results</div>', unsafe_allow_html=True)

        DARK_BG, PLOT_BG, TEXT_C = "#0f1117", "#13202f", "#c8d6e5"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor(DARK_BG)
        for ax in [ax1, ax2]:
            ax.set_facecolor(PLOT_BG)
            ax.tick_params(colors=TEXT_C, labelsize=9)
            for sp in ax.spines.values(): sp.set_color("#1e2a3a")

        scores = [results[0][2], results[1][2]]
        probs  = [results[0][1]*100, results[1][1]*100]
        sc_c   = [("#1e8449" if s>=65 else "#d68910" if s>=35 else "#c0392b") for s in scores]
        pr_c   = [("#c0392b" if p>=65 else "#d68910" if p>=35 else "#1e8449") for p in probs]

        bars1 = ax1.bar(["Job A","Job B"], scores, color=sc_c, width=0.4, edgecolor=DARK_BG)
        for b, sc in zip(bars1, scores):
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+1.5, str(sc),
                     ha="center", color=TEXT_C, fontweight="bold", fontsize=12)
        ax1.set_ylim(0,115); ax1.set_title("Trust Score", color=TEXT_C, fontweight="bold")
        ax1.set_ylabel("Score / 100", color=TEXT_C)
        ax1.yaxis.grid(True, color="#1e2a3a", linestyle="--", alpha=0.5); ax1.set_axisbelow(True)

        bars2 = ax2.bar(["Job A","Job B"], probs, color=pr_c, width=0.4, edgecolor=DARK_BG)
        for b, p in zip(bars2, probs):
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+1.5, f"{p:.1f}%",
                     ha="center", color=TEXT_C, fontweight="bold", fontsize=11)
        ax2.set_ylim(0,115); ax2.set_title("Fake Probability %", color=TEXT_C, fontweight="bold")
        ax2.set_ylabel("Probability %", color=TEXT_C)
        ax2.yaxis.grid(True, color="#1e2a3a", linestyle="--", alpha=0.5); ax2.set_axisbelow(True)

        plt.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        ca, cb = st.columns(2)
        for col, (lbl, pf, ts, rsns) in zip([ca, cb], results):
            with col:
                risk     = ("High Risk" if ts<35 else "Medium Risk" if ts<65 else "Low Risk")
                icon     = {"High Risk":"🚨","Medium Risk":"⚠️","Low Risk":"✅"}[risk]
                card_cls = {"High Risk":"result-high","Medium Risk":"result-medium","Low Risk":"result-low"}[risk]
                ts_color = ("#1e8449" if ts>=65 else "#d68910" if ts>=35 else "#c0392b")
                st.markdown(f"""
                <div class="result-card {card_cls}" style="margin-bottom:12px">
                  <div style="font-size:0.8rem;color:#7f9ab0;margin-bottom:6px">{lbl}</div>
                  <div class="result-icon">{icon}</div>
                  <div class="result-label">{risk}</div>
                  <div style="font-size:1.8rem;font-weight:800;color:{ts_color};margin-top:8px">
                    {ts}<span style="font-size:1rem;color:#7f9ab0">/100</span></div>
                  <div class="result-sub">Trust Score &nbsp;·&nbsp; Fake: {pf*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
                badge_map = {"red":"rbadge-red","orange":"rbadge-orange","green":"rbadge-green"}
                badges = " ".join([f'<span class="rbadge {badge_map[c]}">{txt}</span>' for txt,c in rsns])
                st.markdown(f'<div class="badge-wrap">{badges}</div>', unsafe_allow_html=True)

        winner_idx = 0 if results[0][2] > results[1][2] else 1
        st.markdown("---")
        st.success(f"✅ **{results[winner_idx][0]}** appears more trustworthy — Trust Score: **{results[winner_idx][2]}/100**")


# ══════════════════════════════════════════════════
# PAGE 4 – BATCH ANALYZER
# ══════════════════════════════════════════════════
elif page == "📦 Batch Analyzer":
    st.markdown('<div class="section-title">📦 Batch Job Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload a CSV of job postings and get instant fraud scores for the entire batch</div>', unsafe_allow_html=True)

    st.markdown("""
    > 📌 **CSV must have these columns:**  
    > `title`, `company_profile`, `description`, `requirements`, `salary_range`,
    > `employment_type`, `telecommuting`, `has_company_logo`, `has_questions`
    """)

    use_sample = st.button("▶️ Load built-in sample (50 jobs from dataset)")
    uploaded   = st.file_uploader("— or upload your own CSV file —", type=["csv"])

    df_batch = None
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_batch)} rows from uploaded file.")
    elif use_sample:
        df_batch = df_all.sample(50, random_state=42).reset_index(drop=True)
        st.success("✅ Loaded 50 sample job postings from the built-in dataset.")

    if df_batch is not None:
        st.markdown("---")
        for col in ["company_profile","description","requirements","salary_range","employment_type","title"]:
            if col not in df_batch.columns: df_batch[col] = ""
        for col in ["telecommuting","has_company_logo","has_questions"]:
            if col not in df_batch.columns: df_batch[col] = 0
        df_batch = df_batch.fillna("")

        scores_l, risks_l, probs_l = [], [], []
        prog = st.progress(0, text="Analyzing jobs...")
        for i, (_, row) in enumerate(df_batch.iterrows()):
            feats, vals = build_features(
                str(row["title"]), str(row["company_profile"]),
                str(row["description"]), str(row["requirements"]),
                str(row["salary_range"]), str(row["employment_type"]),
                bool(int(float(row["telecommuting"]))),
                bool(int(float(row["has_company_logo"]))),
                bool(int(float(row["has_questions"])))
            )
            pf   = model.predict_proba(feats)[0][1]
            ts   = compute_trust(vals)
            risk = "High Risk" if ts<35 else "Medium Risk" if ts<65 else "Low Risk"
            scores_l.append(ts); risks_l.append(risk); probs_l.append(round(pf*100,1))
            prog.progress((i+1)/len(df_batch), text=f"Analyzing job {i+1}/{len(df_batch)}...")

        prog.empty()
        df_batch["Trust Score"]      = scores_l
        df_batch["Risk Level"]       = risks_l
        df_batch["Fake Probability"] = probs_l

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Analyzed",  len(df_batch))
        m2.metric("High Risk",       (df_batch["Risk Level"]=="High Risk").sum())
        m3.metric("Medium Risk",     (df_batch["Risk Level"]=="Medium Risk").sum())
        m4.metric("Avg Trust Score", f"{df_batch['Trust Score'].mean():.0f}/100")

        st.markdown("---")
        DARK_BG, PLOT_BG, TEXT_C = "#0f1117", "#13202f", "#c8d6e5"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor(DARK_BG)
        for ax in [ax1, ax2]:
            ax.set_facecolor(PLOT_BG); ax.tick_params(colors=TEXT_C, labelsize=8)
            for sp in ax.spines.values(): sp.set_color("#1e2a3a")

        rc = df_batch["Risk Level"].value_counts()
        rc_c = [{"Low Risk":"#2ecc71","Medium Risk":"#f39c12","High Risk":"#e74c3c"}.get(r,"#7f8c8d") for r in rc.index]
        ax1.bar(rc.index, rc.values, color=rc_c, edgecolor=DARK_BG)
        ax1.set_title("Risk Level Distribution", color=TEXT_C, fontweight="bold")
        ax1.set_ylabel("Count", color=TEXT_C)
        ax1.yaxis.grid(True, color="#1e2a3a", linestyle="--", alpha=0.5); ax1.set_axisbelow(True)

        ax2.hist(df_batch["Trust Score"], bins=20, color="#2980b9", edgecolor=DARK_BG, alpha=0.9)
        ax2.set_title("Trust Score Distribution", color=TEXT_C, fontweight="bold")
        ax2.set_xlabel("Trust Score", color=TEXT_C)
        ax2.yaxis.grid(True, color="#1e2a3a", linestyle="--", alpha=0.5); ax2.set_axisbelow(True)

        plt.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("**📋 Full Results Table**")
        display_cols = ["title","company_profile","Trust Score","Risk Level","Fake Probability"]
        if "fraudulent" in df_batch.columns:
            display_cols.append("fraudulent")
        st.dataframe(
            df_batch[display_cols].rename(columns={"company_profile":"Company","fraudulent":"Actual Fraud"}),
            use_container_width=True
        )
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=df_batch[display_cols].to_csv(index=False),
            file_name="fraudlens_batch_results.csv",
            mime="text/csv"
        )
