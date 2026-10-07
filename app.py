"""
NEXUS: Workforce Transition Intelligence
Main Streamlit Application — State-Driven Traversal

Traversable Journey:
  Step 1: Identity & Verified Evidence Ingestion
  Step 2: Verified Capability Profile
  Step 3: Market Fit & Positions
  Step 4: Transition Bottleneck Analysis
  Step 5: Counterfactual What-If Simulation
  Step 6: Scientific Audit & Methodology
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timezone

# Streamlit Page Configuration
st.set_page_config(
    page_title="NEXUS | Career & Transition Intelligence",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Backend modules
from backend.data_loader import (
    load_all_raw_datasets, get_settings, get_skill_aliases,
    get_interventions, get_demo_candidate
)
from backend.data_cleaner import DataCleaner
from backend.skill_analysis import SkillAnalyzer
from backend.market_analysis import MarketAnalyzer
from backend.success_analysis import SuccessAnalyzer
from backend.opportunity_engine import OpportunityEngine
from backend.simulation import TransitionSimulator
from backend.database import CandidateDB
from backend.evidence_ingestion import (
    ResumeConnector, GitHubConnector, LeetCodeConnector,
    SkillPlatformConnector, infer_usernames_from_email
)
from backend.evidence_scoring import EvidenceScorer

# Frontend modules
from frontend.styles import (
    CUSTOM_CSS, render_kpi, render_micro_explanation
)
from frontend.ui import (
    plot_coefficients_bar,
    plot_confusion_matrix_heatmap,
    plot_capability_radar,
    plot_bottlenecks_bar,
    plot_three_futures_comparison
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Custom Styles for Step Navigation and Cards
st.markdown("""
<style>
.nexus-top-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0284c7 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 16px;
    color: white;
}
.profile-badge {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 10px;
    padding: 10px 16px;
    text-align: right;
}
.step-nav-bar {
    display: flex;
    gap: 8px;
    margin-bottom: 24px;
    background: #f1f5f9;
    padding: 8px;
    border-radius: 12px;
}
.step-header-box {
    margin: 12px 0 20px 0;
}
.step-header-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #0f172a;
}
.step-header-desc {
    font-size: 0.92rem;
    color: #64748b;
    margin-top: 4px;
}
.status-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
}
.status-pill.success { background: #dcfce7; color: #166534; }
.status-pill.info { background: #e0f2fe; color: #0369a1; }
.status-pill.warning { background: #fef3c7; color: #92400e; }

.offer-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
    border: 2px solid #86efac;
    border-radius: 12px;
    padding: 22px 26px;
    margin: 16px 0 24px 0;
}
.offer-val-lg {
    font-size: 2.2rem;
    font-weight: 900;
    color: #166534;
    line-height: 1.1;
}
.curriculum-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 4px solid #0284c7;
}
.curriculum-module-title {
    font-weight: 800;
    color: #0f172a;
    font-size: 0.96rem;
    margin-bottom: 4px;
}
.curriculum-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    background: #e0f2fe;
    color: #0369a1;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Global Cached Pipeline & Database Initialization
# ----------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_nexus_system():
    raw_data = load_all_raw_datasets()
    cleaner = DataCleaner()
    cleaned_data, quality_reports = cleaner.clean_all(raw_data)

    market_analyzer = MarketAnalyzer(cleaned_data["analytics_jobs"], cleaned_data["datascience_jobs"])
    headline_metrics = market_analyzer.get_headline_metrics()
    role_benchmarks = market_analyzer.get_role_demand_and_salary_benchmarks()

    skill_analyzer = SkillAnalyzer()
    skill_freqs = skill_analyzer.compute_skill_frequencies(cleaned_data["analytics_jobs"])
    role_profiles = skill_analyzer.compute_role_skill_matrix(cleaned_data["analytics_jobs"])

    success_analyzer = SuccessAnalyzer(cleaned_data["jds_skills"], cleaned_data["sds_personality"])
    jds_insights = success_analyzer.analyze_jds()
    sds_insights = success_analyzer.analyze_sds()

    opp_engine = OpportunityEngine(role_profiles, role_benchmarks, skill_freqs)
    simulator = TransitionSimulator(opp_engine)

    db = CandidateDB()
    db.seed_demo_candidate_if_missing()

    return {
        "cleaner": cleaner,
        "cleaned_data": cleaned_data,
        "quality_reports": quality_reports,
        "market_analyzer": market_analyzer,
        "headline_metrics": headline_metrics,
        "role_benchmarks": role_benchmarks,
        "skill_analyzer": skill_analyzer,
        "skill_freqs": skill_freqs,
        "role_profiles": role_profiles,
        "success_analyzer": success_analyzer,
        "jds_insights": jds_insights,
        "sds_insights": sds_insights,
        "opp_engine": opp_engine,
        "simulator": simulator,
        "db": db
    }

# Styling for spinners and loading animations
st.markdown("""
<style>
@keyframes nexus-pulse {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.35; }
    40%            { transform: scale(1.0);  opacity: 1.0; }
}
@keyframes nexus-bar {
    0%   { width: 0%;   opacity: 0.8; }
    50%  { width: 75%;  opacity: 1.0; }
    100% { width: 100%; opacity: 0.85; }
}
.stSpinner > div {
    border-top-color: #0284c7 !important;
    border-right-color: #38bdf8 !important;
    animation-duration: 0.75s !important;
}
.nexus-loader-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 60px 20px;
}
.nexus-loader-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0284c7 100%);
    border-radius: 18px;
    padding: 40px 50px;
    text-align: center;
    box-shadow: 0 20px 50px rgba(2,132,199,0.3);
    max-width: 440px;
    width: 100%;
}
.nexus-loader-title {
    font-size: 1.8rem;
    font-weight: 900;
    color: #ffffff;
    letter-spacing: 2px;
    margin-bottom: 4px;
}
.nexus-loader-sub {
    font-size: 0.88rem;
    color: #93c5fd;
    margin-bottom: 24px;
}
.nexus-dots {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-bottom: 22px;
}
.nexus-dot {
    width: 13px; height: 13px;
    border-radius: 50%;
    background: #38bdf8;
    animation: nexus-pulse 1.3s ease-in-out infinite;
}
.nexus-dot:nth-child(2) { animation-delay: 0.2s; background: #7dd3fc; }
.nexus-dot:nth-child(3) { animation-delay: 0.4s; background: #bae6fd; }
.nexus-dot:nth-child(4) { animation-delay: 0.6s; background: #e0f2fe; }
.nexus-progress-wrap {
    background: rgba(255,255,255,0.18);
    border-radius: 8px;
    height: 6px;
    overflow: hidden;
    margin: 0 auto 16px auto;
    width: 240px;
}
.nexus-progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #38bdf8, #0ea5e9, #38bdf8);
    border-radius: 8px;
    animation: nexus-bar 2.8s ease-out infinite;
}
.nexus-loader-msg {
    font-size: 0.82rem;
    color: #bae6fd;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

_init_loader_placeholder = st.empty()
_init_loader_placeholder.markdown("""
<div class="nexus-loader-wrap">
  <div class="nexus-loader-card">
    <div class="nexus-loader-title">🧭 NEXUS</div>
    <div class="nexus-loader-sub">Workforce Transition Intelligence</div>
    <div class="nexus-dots">
      <div class="nexus-dot"></div>
      <div class="nexus-dot"></div>
      <div class="nexus-dot"></div>
      <div class="nexus-dot"></div>
    </div>
    <div class="nexus-progress-wrap">
      <div class="nexus-progress-bar"></div>
    </div>
    <div class="nexus-loader-msg">⚡ Initializing market intelligence & verified models...</div>
  </div>
</div>
""", unsafe_allow_html=True)

SYSTEM = initialize_nexus_system()
DB: CandidateDB = CandidateDB()
_init_loader_placeholder.empty()

# ----------------------------------------------------
# Session State
# ----------------------------------------------------
def _init_state():
    defaults = {
        "current_step": 1,
        "user_email": "",
        "user_name": "",
        "user_title": "Data Practitioner",
        "user_exp": 0.0,
        "candidate_id": "",
        "candidate_info": {},
        "derived_capabilities": {},    # EvidenceScorer keys — for display in Step 2
        "engine_capabilities": {},     # Role-profile keys — for OpportunityEngine/Simulator
        "provenance_reasons": {},
        "confidence_scores": {},
        "discovered_gh_matches": [],
        "selected_gh_user": "",
        "selected_lc_user": "",
        "fetch_feedback": [],
        "opt_in_discoverable": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

SKILL_DISPLAY_NAMES = {
    "python": "Python & Data Wrangling",
    "sql": "SQL & Relational Databases",
    "machine_learning": "Machine Learning & Modeling",
    "statistics_math": "Statistics & Quantitative Math",
    "business_intelligence": "Business Intelligence & BI Tools",
    "data_storytelling": "Data Storytelling & Communication",
    "big_data_cloud": "Big Data & Cloud Infrastructure",
    "deep_learning_nlp": "Deep Learning & NLP",
    "data_engineering": "Data Engineering & Pipelines",
    "mlops_production": "MLOps & Production Deployment"
}

def format_experience_label(exp_val: float) -> str:
    total_months = int(round(float(exp_val) * 12))
    yrs = total_months // 12
    mos = total_months % 12
    if yrs == 0 and mos == 0:
        return "0 yrs (Fresher / Student)"
    parts = []
    if yrs > 0:
        parts.append(f"{yrs} yr{'s' if yrs != 1 else ''}")
    if mos > 0:
        parts.append(f"{mos} mo{'s' if mos != 1 else ''}")
    return " ".join(parts)

# -----------------------------------------------------------------------
# Key translation: EvidenceScorer canonical → OpportunityEngine role keys
# These are the keys used in role_profiles (SkillAnalyzer) and
# interventions.json. ALL engine calls must use ENGINE_KEY_MAP output.
# -----------------------------------------------------------------------
ENGINE_KEY_MAP = {
    # EvidenceScorer key        → role_profile / intervention key
    "python":                   "python",
    "sql":                      "sql_database",
    "machine_learning":         "machine_learning",
    "statistics_math":          "math_statistics",
    "business_intelligence":    "business_analysis",
    "data_storytelling":        "visualization_storytelling",
    "big_data_cloud":           "big_data_engineering",
    "deep_learning_nlp":        "machine_learning",   # map deep learning into ML bucket
    "data_engineering":         "big_data_engineering",
    "mlops_production":         "cloud_deployment_mlops",
}

def translate_to_engine_caps(display_caps: dict) -> dict:
    """
    Translates EvidenceScorer canonical keys → OpportunityEngine role_profile keys.
    When multiple display keys map to the same engine key, takes the max value.
    """
    engine_caps = {}
    for d_key, e_key in ENGINE_KEY_MAP.items():
        val = display_caps.get(d_key, 0.20)
        engine_caps[e_key] = max(engine_caps.get(e_key, 0.0), val)
    return engine_caps

def get_role_benchmark_data(role_name: str) -> dict:
    df_b = SYSTEM["role_benchmarks"]
    match = df_b[df_b["job_title_clean"].str.lower() == role_name.strip().lower()]
    if not match.empty:
        r = match.iloc[0]
        return {
            "avg_salary": float(r["avg_salary_lakhs"]),
            "min_salary": float(r["min_salary_lakhs"]),
            "max_salary": float(r["max_salary_lakhs"]),
            "vacancies": int(r["total_vacancies"]),
            "req_exp": float(r["min_experience_years"])
        }
    return {
        "avg_salary": 12.0,
        "min_salary": 7.0,
        "max_salary": 19.0,
        "vacancies": 1000,
        "req_exp": 2.0
    }

def compute_expected_offer(role_name: str, compat_pct: float, exp_years: float) -> dict:
    bdata = get_role_benchmark_data(role_name)
    min_sal = bdata["min_salary"]
    avg_sal = bdata["avg_salary"]
    max_sal = bdata["max_salary"]
    req_exp = bdata["req_exp"]

    exp_factor = min(1.0, max(exp_years, 0.5) / max(req_exp, 0.5))
    s = min(1.0, max(0.0, compat_pct / 100.0))

    if s < 0.50:
        status = "Blocked"
        expected = min_sal * max(0.70, (s / 0.50) * 0.90)
    elif s < 0.65:
        status = "Stretch"
        norm = (s - 0.50) / 0.15
        expected = min_sal + norm * (avg_sal - min_sal) * (0.65 + 0.35 * exp_factor)
    else:
        status = "Reachable"
        norm = min(1.0, (s - 0.65) / 0.35)
        expected = avg_sal + norm * (max_sal - avg_sal) * (0.50 + 0.50 * exp_factor)

    expected = max(min_sal * 0.65, min(max_sal, expected))
    return {
        "expected_salary": round(expected, 2),
        "min_salary": round(min_sal, 2),
        "avg_salary": round(avg_sal, 2),
        "max_salary": round(max_sal, 2),
        "status": status,
        "vacancies": bdata["vacancies"],
        "req_exp": req_exp
    }

INTERVENTION_CURRICULUM = {
    "INTERVENTION A": {
        "title": "Path A: Advanced AI/ML & Deep Learning Curriculum",
        "category": "Machine Learning & Artificial Intelligence",
        "duration": "10 - 12 Weeks (8-10 hrs/week)",
        "skills_boosted": [
            ("machine_learning", "Machine Learning & Predictive Modeling", "+35%"),
            ("python", "Python & Vectorized Processing", "+15%"),
            ("math_statistics", "Quantitative Math & Statistics", "+15%")
        ],
        "key_tools": ["PyTorch", "scikit-learn", "XGBoost", "Optuna", "MLflow", "FastAPI", "Docker"],
        "modules": [
            {
                "num": 1,
                "name": "Predictive Modeling & Statistical Learning",
                "topics": "L1/L2 ElasticNet regularization, gradient boosted decision trees (XGBoost/LightGBM), hyperparameter optimization with Optuna, handling severe class imbalance with SMOTE and focal loss."
            },
            {
                "num": 2,
                "name": "Deep Learning & Neural Architectures (PyTorch)",
                "topics": "Deep backprop math, PyTorch tensors & custom datasets, CNNs for feature representation, attention mechanisms, fine-tuning transformer pipelines with HuggingFace."
            },
            {
                "num": 3,
                "name": "Production ML Inference & MLOps Pipelines",
                "topics": "Model serialization (ONNX/TorchScript), low-latency REST inference APIs with FastAPI, automated drift monitoring, MLflow experiment registry."
            }
        ],
        "capstone": "Enterprise Predictive Service: Train a high-performing gradient boosted model and deep transformer on customer tabular/text data, package inference into a containerized FastAPI service with automated CI testing.",
        "target_roles": ["Data Scientist", "Machine Learning Engineer", "Senior Data Scientist"]
    },
    "INTERVENTION B": {
        "title": "Path B: Analytics Translation & Executive Storytelling Curriculum",
        "category": "BI & Strategic Data Storytelling",
        "duration": "6 - 8 Weeks (6-8 hrs/week)",
        "skills_boosted": [
            ("visualization_storytelling", "Data Visualization & Storytelling", "+30%"),
            ("business_analysis", "Business Problem Formulation & KPIs", "+25%"),
            ("sas", "SAS & Advanced Statistical Analysis", "+20%")
        ],
        "key_tools": ["Power BI", "Tableau", "DAX", "SQL Window Functions", "Financial Modeling", "Executive Storytelling"],
        "modules": [
            {
                "num": 1,
                "name": "Executive Dashboard Architecture",
                "topics": "Star schema dimensional data modeling, optimized DAX measures (Time intelligence, window calculations), user-centric layout design, responsive interactive drill-throughs."
            },
            {
                "num": 2,
                "name": "Business Metric Formulation & Unit Economics",
                "topics": "Customer Lifetime Value (LTV), Customer Acquisition Cost (CAC), Cohort Retention curves, Net Revenue Retention (NRR), statistical A/B test interpretation for product teams."
            },
            {
                "num": 3,
                "name": "The Pyramid Principle & Boardroom Storytelling",
                "topics": "Structuring executive insight memos using Barbara Minto's Pyramid Principle, translating p-values and regression outputs into financial EBITDA ROI, crafting compelling C-Suite slide decks."
            }
        ],
        "capstone": "Executive Commercial Performance Suite: Design a published interactive Power BI/Tableau suite synthesizing 100,000+ transaction rows with automated diagnostic insights and a 5-page PDF executive decision memo.",
        "target_roles": ["Business Analyst", "Senior Data Analyst", "Senior Business Analyst"]
    },
    "INTERVENTION C": {
        "title": "Path C: Production Data Engineering & Cloud MLOps Curriculum",
        "category": "Data Systems & Cloud Infrastructure",
        "duration": "10 - 12 Weeks (8-10 hrs/week)",
        "skills_boosted": [
            ("big_data_engineering", "Big Data & Distributed Computing", "+35%"),
            ("cloud_deployment_mlops", "Cloud Deployment & MLOps", "+35%"),
            ("sql_database", "Advanced SQL & Relational Architecture", "+15%")
        ],
        "key_tools": ["Apache Spark (PySpark)", "Apache Kafka", "Snowflake / BigQuery", "dbt", "Apache Airflow", "Docker", "AWS/GCP"],
        "modules": [
            {
                "num": 1,
                "name": "Distributed Data Processing (PySpark & Kafka)",
                "topics": "Spark cluster execution engine, broadcast joins, data partitioning strategies, Spark SQL optimization, real-time message streaming with Apache Kafka."
            },
            {
                "num": 2,
                "name": "Modern Data Stack & Warehouse Modeling",
                "topics": "Cloud data warehousing (Snowflake / Google BigQuery), Slowly Changing Dimensions (SCD Type 2), automated modular transformations using dbt (Data Build Tool)."
            },
            {
                "num": 3,
                "name": "Workflow Orchestration & Cloud Infrastructure",
                "topics": "Directed Acyclic Graph (DAG) construction in Apache Airflow, Docker containerized services, CI/CD automated deployment pipelines with GitHub Actions."
            }
        ],
        "capstone": "Scalable Cloud Data Platform: Implement an end-to-end data pipeline extracting real-time streaming data via Kafka, transforming with PySpark and dbt in Snowflake, and orchestrating hourly runs with Apache Airflow.",
        "target_roles": ["Data Engineer", "Senior Data Engineer", "Data Architect"]
    }
}

def recalculate_profile(candidate_id: str):
    events = DB.get_skill_events_for_candidate(candidate_id)
    scorer = EvidenceScorer()
    scored = scorer.score_candidate_events(events)

    # Store display caps (EvidenceScorer canonical keys) — used for profile display
    st.session_state.derived_capabilities = scored["capabilities"]
    st.session_state.confidence_scores = scored["confidence_scores"]
    st.session_state.provenance_reasons = scored["provenance_explanations"]

    # Store engine caps (role_profile keys) — used for OpportunityEngine & Simulator
    st.session_state.engine_capabilities = translate_to_engine_caps(scored["capabilities"])

    DB.save_derived_skills(
        candidate_id,
        st.session_state.derived_capabilities,
        st.session_state.confidence_scores,
        st.session_state.provenance_reasons
    )

def reset_to_new_profile():
    # 1. Reset domain state
    st.session_state.user_email = ""
    st.session_state.user_name = ""
    st.session_state.user_title = "Data Practitioner"
    st.session_state.user_exp = 0.0
    st.session_state.candidate_id = ""
    st.session_state.candidate_info = {}
    st.session_state.derived_capabilities = {}
    st.session_state.engine_capabilities = {}
    st.session_state.provenance_reasons = {}
    st.session_state.confidence_scores = {}
    st.session_state.discovered_gh_matches = []
    st.session_state.selected_gh_user = ""
    st.session_state.selected_lc_user = ""
    st.session_state.fetch_feedback = []
    st.session_state.current_step = 1

    # 2. Clear all Streamlit widget state keys so all input fields are completely blank
    widget_keys = [
        "step1_email", "step1_name", "step1_exp", "step1_exp_years", "step1_exp_months",
        "exact_gh_user", "exact_lc_user", "step1_resume", "step1_fresh_audit",
        "step5_target_role_select", "step5_inv_select"
    ]
    for wk in widget_keys:
        if wk in st.session_state:
            del st.session_state[wk]
    for k in list(st.session_state.keys()):
        if k.startswith("slider_step1_"):
            del st.session_state[k]

# ----------------------------------------------------
# Sidebar Menu: Global Navigation & Profile Management
# ----------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 12px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 12px;">
        <h2 style="margin: 0; font-size: 1.45rem; font-weight: 900; color: #0f172a;">🧭 NEXUS</h2>
        <div style="color: #0284c7; font-size: 0.82rem; font-weight: 600;">Workforce Transition Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Active Candidate Status Card
    if st.session_state.user_email:
        name_show = st.session_state.user_name or st.session_state.user_email.split('@')[0].title()
        exp_show = format_experience_label(st.session_state.user_exp)
        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 12px; margin-bottom: 12px;">
            <div style="font-weight: 800; color: #0f172a; font-size: 0.95rem;">👤 {name_show}</div>
            <div style="font-size: 0.78rem; color: #0284c7; word-break: break-all;">✉️ {st.session_state.user_email}</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 3px;">💼 Exp: <strong>{exp_show}</strong></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #fffbeb; border: 1px solid #fef3c7; border-radius: 8px; padding: 10px 12px; margin-bottom: 12px;">
            <div style="font-weight: 700; color: #92400e; font-size: 0.85rem;">⚠️ No Profile Loaded</div>
            <div style="font-size: 0.75rem; color: #b45309;">Enter your details in Step 1 to begin.</div>
        </div>
        """, unsafe_allow_html=True)

    # Create New Person Profile Action
    if st.button("➕ Create New Profile", use_container_width=True):
        reset_to_new_profile()
        st.rerun()

    st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.85rem; font-weight: 800; color: #475569; margin-bottom: 6px;'>📍 PROCESS STEPS</div>", unsafe_allow_html=True)

    steps_meta = [
        (1, "👤 1. Identity & Evidence"),
        (2, "📊 2. Capabilities Profile"),
        (3, "🎯 3. Market Fit & Roles"),
        (4, "⚠️ 4. Bottleneck Gap"),
        (5, "🔮 5. What-If Simulation"),
        (6, "🔬 6. Scientific Audit")
    ]

    for step_num, step_label in steps_meta:
        is_active = (st.session_state.current_step == step_num)
        btn_type = "primary" if is_active else "secondary"
        if st.button(step_label, key=f"sidebar_step_btn_{step_num}", type=btn_type, use_container_width=True):
            st.session_state.current_step = step_num
            st.rerun()

    st.markdown("<hr style='margin: 16px 0 10px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.76rem; color: #64748b; line-height: 1.4;">
        <strong>Market Evidence:</strong><br>
        • 15,841 real job postings<br>
        • 10 analyzed role families<br>
        • L2-regularized logistic models<br>
        • Counterfactual simulation
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# Banner Header & Persistent Identity
# ----------------------------------------------------
banner_col1, banner_col2 = st.columns([7, 3])
with banner_col1:
    st.markdown("""
    <div style="padding: 6px 0;">
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 900; color: #0f172a;">🧭 NEXUS</h1>
        <div style="color: #0284c7; font-weight: 600; font-size: 0.95rem;">Workforce Transition Intelligence Platform</div>
        <div style="color: #64748b; font-size: 0.82rem;">15,841 real postings · L2-Regularized Logistic Models · Counterfactual Simulation</div>
    </div>
    """, unsafe_allow_html=True)

with banner_col2:
    if st.session_state.user_email:
        name_show = st.session_state.user_name or st.session_state.user_email.split('@')[0].title()
        st.markdown(f"""
        <div class="identity-card" style="margin: 0; padding: 10px 14px; text-align: right;">
            <div style="font-size: 1rem; font-weight: 800; color: #0c4a6e;">👤 {name_show}</div>
            <div style="font-size: 0.8rem; color: #0284c7;">✉️ {st.session_state.user_email}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: right; padding-top: 15px;">
            <span class="status-pill warning">No Profile Loaded</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr style='margin: 12px 0 16px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

# ----------------------------------------------------
# Global Step Navigation Bar (Traverse to ANY step)
# ----------------------------------------------------
col_nav = st.columns(6)
for col, (step_num, step_label) in zip(col_nav, steps_meta):
    is_active = (st.session_state.current_step == step_num)
    btn_type = "primary" if is_active else "secondary"
    if col.button(step_label, key=f"global_nav_step_{step_num}", type=btn_type, use_container_width=True):
        st.session_state.current_step = step_num
        st.rerun()

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)


# =====================================================================
# STEP 1: IDENTITY & EVIDENCE
# =====================================================================
if st.session_state.current_step == 1:
    st.markdown("""
    <div class="step-header-box">
        <div class="step-header-title">Step 1: Identity & Verified Evidence Ingestion</div>
        <div class="step-header-desc">Enter your email and exact public profile handles (GitHub, LeetCode) or upload your resume. Zero filler or mock data is generated.</div>
    </div>
    """, unsafe_allow_html=True)

    col_e1, col_e2 = st.columns([5, 5])
    with col_e1:
        st.markdown("#### 1. Enter Your Details")
        email_val = st.text_input("Email address (required):", value=st.session_state.user_email, placeholder="e.g. aryan.rai@gmail.com", key="step1_email")
        name_val = st.text_input("Full name (optional):", value=st.session_state.user_name, placeholder="e.g. Aryan Rai", key="step1_name")

        # Experience Gradient: Years & Months (Defaults to 0 yrs, 0 mos)
        st.markdown("<label style='font-size: 0.88rem; font-weight: 600; color: #334155; margin-bottom: 2px; display: block;'>Work Experience:</label>", unsafe_allow_html=True)
        col_exp_y, col_exp_m = st.columns(2)
        cur_exp = float(st.session_state.get("user_exp", 0.0))
        cur_yrs = int(cur_exp)
        cur_mos = int(round((cur_exp - cur_yrs) * 12))
        if cur_mos >= 12:
            cur_yrs += 1
            cur_mos = 0

        with col_exp_y:
            exp_years = st.number_input("Years:", min_value=0, max_value=30, value=cur_yrs, step=1, key="step1_exp_years")
        with col_exp_m:
            exp_months = st.number_input("Months:", min_value=0, max_value=11, value=cur_mos, step=1, key="step1_exp_months")

        exp_val = round(float(exp_years) + (float(exp_months) / 12.0), 2)
        st.session_state.user_exp = exp_val

        exp_badge = format_experience_label(exp_val)
        st.caption(f"⏱️ Total Experience: **{exp_badge}** ({exp_val:.2f} yrs)")

        col_demo_btn, col_new_btn = st.columns(2)
        with col_demo_btn:
            if st.button("⚡ Load Demo Candidate", use_container_width=True):
                cand = DB.seed_demo_candidate_if_missing()
                st.session_state.user_email = cand["email"]
                st.session_state.user_name = cand["name"]
                st.session_state.user_exp = cand["years_of_experience"]
                st.session_state.candidate_id = cand["candidate_id"]
                st.session_state.candidate_info = cand
                recalculate_profile(cand["candidate_id"])
                st.session_state.fetch_feedback = ["Loaded verified competition portfolio for Aarav Sharma (10 evidence records)."]
                st.session_state.current_step = 2
                st.rerun()

        with col_new_btn:
            if st.button("➕ New Profile", use_container_width=True):
                reset_to_new_profile()
                st.rerun()

    with col_e2:
        st.markdown("#### 2. Enter Public Profiles to Audit (Optional)")
        st.caption("Enter your exact public handles to pull verified contributions and metrics directly:")

        gh_input_user = st.text_input(
            "GitHub Username to audit (optional):",
            value=st.session_state.selected_gh_user,
            placeholder="e.g. AryanRai or octocat (leave blank if none)",
            key="exact_gh_user"
        )
        st.session_state.selected_gh_user = gh_input_user.strip()

        lc_input_user = st.text_input(
            "LeetCode Username to audit (optional):",
            value=st.session_state.selected_lc_user,
            placeholder="e.g. AryanRai (leave blank if none)",
            key="exact_lc_user"
        )
        st.session_state.selected_lc_user = lc_input_user.strip()

        st.markdown("""
        <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 12px; margin-top: 16px; font-size: 0.82rem; color: #475569; line-height: 1.45;">
            💡 <strong>Unified Ingestion:</strong> Enter your handles above and any optional evidence below (Resume, Coursework, or Self-Assessment).
            Then click the single <strong>Proceed to Step 2</strong> button at the bottom to process all inputs together.
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.fetch_feedback:
        st.markdown("---")
        st.markdown("##### Ingestion Activity Log:")
        for fb in st.session_state.fetch_feedback:
            st.write(fb)

    st.markdown("---")
    st.markdown("#### 3. Additional Evidence Sources & Cold-Start Seeding (Optional)")
    tab_res, tab_coursework, tab_manual, tab_ledger = st.tabs([
        "📄 Upload Resume",
        "🎓 Coursework & Syllabus (Cold Start)",
        "✍️ Self-Declared Skills",
        "🗄️ Evidence Ledger"
    ])

    with tab_res:
        uploaded_resume = st.file_uploader("Upload resume (PDF / DOCX):", type=["pdf", "docx"], key="step1_resume")
        if uploaded_resume:
            st.info(f"📄 Resume **{uploaded_resume.name}** selected. It will be audited and ingested when you click the button below.")

    with tab_coursework:
        st.markdown("##### 🎓 Cold Start: University Syllabi, Coursework & Academic Labs")
        st.caption("No public coding portfolio or industry experience yet? Seed your initial verified capability baseline directly from completed university coursework, engineering labs, and academic assessments (Build for Bharat Slide 01 & 06).")

        col_cw1, col_cw2 = st.columns(2)
        with col_cw1:
            cw_institution = st.text_input("University / Institution Name:", placeholder="e.g. Indian Institute of Technology / NIT / Anna University", key="cw_inst_name")
            cw_degree = st.selectbox("Degree / Program Track:", ["B.Tech / B.E. Computer Science & IT", "B.Tech / B.E. Electrical & Electronics", "B.Sc / M.Sc Data Science & Statistics", "BCA / MCA Information Technology", "Other STEM Undergraduate"], key="cw_degree_prog")
            cw_perf = st.selectbox("Academic Performance / Grade Standing:", [
                "Distinction / A Grade (80%+ / 8.5+ CGPA) — Advanced Aptitude",
                "First Class / B Grade (65–79% / 7.0+ CGPA) — Proficient Aptitude",
                "Pass / Second Class (50–64%) — Foundational Aptitude"
            ], key="cw_grade_perf")

        with col_cw2:
            st.markdown("**Select Completed Courses & Labs:**")
            cw_sel_dsa = st.checkbox("Data Structures & Algorithms (Python / C++)", value=True, key="cw_dsa")
            cw_sel_dbms = st.checkbox("Database Management Systems (RDBMS & SQL Labs)", value=True, key="cw_dbms")
            cw_sel_stats = st.checkbox("Probability, Linear Algebra & Applied Statistics", value=True, key="cw_stats")
            cw_sel_ml = st.checkbox("Machine Learning & Applied Modeling Lab", value=False, key="cw_ml")
            cw_sel_bi = st.checkbox("Data Visualization & Business Analytics", value=False, key="cw_bi")
            cw_sel_cloud = st.checkbox("Cloud Computing & Distributed Systems", value=False, key="cw_cloud")
            cw_sel_nlp = st.checkbox("Deep Learning & Natural Language Processing", value=False, key="cw_nlp")

        if st.button("🌱 Register Coursework Signals", key="btn_seed_coursework"):
            clean_email = email_val.strip().lower() or "student@university.ac.in"
            cand = DB.get_or_create_candidate(clean_email, name=name_val.strip() or "Student Scholar", title=cw_degree, years_exp=exp_val)
            st.session_state.candidate_id = cand["candidate_id"]
            st.session_state.candidate_info = cand
            st.session_state.user_email = clean_email
            st.session_state.user_title = cw_degree

            base_score = 0.85 if "Distinction" in cw_perf else (0.72 if "First Class" in cw_perf else 0.60)
            cw_events = []
            if cw_sel_dsa:
                cw_events.append({"skill": "python", "topic": "Data Structures, Algorithms & Problem Solving", "score": base_score, "difficulty": "MEDIUM", "volume": 1, "recency": 3.0})
            if cw_sel_dbms:
                cw_events.append({"skill": "sql", "topic": "Relational Databases, Normalization & SQL Queries", "score": base_score, "difficulty": "MEDIUM", "volume": 1, "recency": 3.0})
            if cw_sel_stats:
                cw_events.append({"skill": "statistics_math", "topic": "Linear Algebra, Probability & Applied Statistical Tests", "score": round(base_score * 0.95, 2), "difficulty": "MEDIUM", "volume": 1, "recency": 3.0})
            if cw_sel_ml:
                cw_events.append({"skill": "machine_learning", "topic": "Supervised/Unsupervised Learning & Model Evaluation", "score": round(base_score * 0.90, 2), "difficulty": "MEDIUM", "volume": 1, "recency": 3.0})
            if cw_sel_bi:
                cw_events.append({"skill": "business_intelligence", "topic": "Business Intelligence & Metric Reporting", "score": round(base_score * 0.85, 2), "difficulty": "EASY", "volume": 1, "recency": 3.0})
                cw_events.append({"skill": "data_storytelling", "topic": "Exploratory Data Analysis & Presentation", "score": round(base_score * 0.85, 2), "difficulty": "EASY", "volume": 1, "recency": 3.0})
            if cw_sel_cloud:
                cw_events.append({"skill": "big_data_cloud", "topic": "Distributed Cloud Systems & Virtualization", "score": round(base_score * 0.80, 2), "difficulty": "MEDIUM", "volume": 1, "recency": 3.0})
            if cw_sel_nlp:
                cw_events.append({"skill": "deep_learning_nlp", "topic": "Neural Networks, Transformers & NLP Pipelines", "score": round(base_score * 0.80, 2), "difficulty": "HARD", "volume": 1, "recency": 3.0})

            if not cw_events:
                st.warning("Please check at least one completed university course.")
            else:
                inst_label = cw_institution.strip() or "University Syllabus"
                src_id = DB.register_source(cand["candidate_id"], "COURSEWORK", f"{inst_label} — {cw_degree}")
                DB.add_evidence(
                    cand["candidate_id"], src_id, "ACADEMIC_SYLLABUS_RECORD",
                    f"University Coursework: {inst_label}",
                    f"Accredited coursework modules in {cw_degree} with standing: {cw_perf}.",
                    "MEDIUM",
                    base_score, 3.0, cw_events
                )
                recalculate_profile(cand["candidate_id"])
                st.success(f"✓ Successfully registered {len(cw_events)} capability signals from {inst_label} coursework! Click the button below to proceed.")

    with tab_manual:
        st.caption("Rate yourself on skills not captured in external profiles:")
        col_m1, col_m2 = st.columns(2)
        half_skills = len(SKILL_DISPLAY_NAMES) // 2
        for idx, (sk, s_name) in enumerate(SKILL_DISPLAY_NAMES.items()):
            col_target = col_m1 if idx < half_skills else col_m2
            with col_target:
                slider_opts = [0.20, 0.40, 0.60, 0.80, 0.95]
                raw_sk_val = float(st.session_state.derived_capabilities.get(sk, 0.20))
                closest_opt = min(slider_opts, key=lambda x: abs(x - raw_sk_val))
                lvl = st.select_slider(
                    s_name,
                    options=slider_opts,
                    value=closest_opt,
                    format_func=lambda x: {0.20: "None/Beginner", 0.40: "Developing", 0.60: "Proficient", 0.80: "Advanced", 0.95: "Mastery"}.get(x, str(x)),
                    key=f"slider_step1_{sk}"
                )
                if lvl > 0.25 and st.session_state.candidate_id:
                    st.session_state.derived_capabilities[sk] = lvl

        if st.button("💾 Save Declared Skills", key="btn_save_manual_step1"):
            clean_email = email_val.strip().lower() or "candidate@local.internal"
            cand = DB.get_or_create_candidate(clean_email, name=name_val.strip() or "Candidate", title="Data Practitioner", years_exp=exp_val)
            st.session_state.candidate_id = cand["candidate_id"]
            st.session_state.candidate_info = cand
            st.session_state.user_email = clean_email
            src_id = DB.register_source(st.session_state.candidate_id, "SELF_DECLARED", "Self_Assessment")
            events = [
                {"skill": k, "topic": "Self Assessment", "score": v, "difficulty": "EASY", "volume": 1, "recency": 1.0}
                for k, v in st.session_state.derived_capabilities.items() if v > 0.25
            ]
            DB.add_evidence(
                st.session_state.candidate_id, src_id, "SELF_DECLARED_RECORD",
                "Self-Assessment", "User-declared proficiency.", "WEAK", 0.60, 1.0, events
            )
            recalculate_profile(st.session_state.candidate_id)
            st.success("✓ Saved self-declared skills into Evidence Ledger! Click the button below to proceed.")

    with tab_ledger:
        if st.session_state.candidate_id:
            summary = DB.get_candidate_evidence_summary(st.session_state.candidate_id)
            if summary:
                st.dataframe(pd.DataFrame(summary)[["title", "source_type", "verification_strength", "recency_months", "timestamp"]].rename(columns={
                    "title": "Evidence Title", "source_type": "Source Type", "verification_strength": "Strength",
                    "recency_months": "Recency (Mo)", "timestamp": "Timestamp"
                }), width="stretch")
                col_cl1, col_cl2 = st.columns([6, 4])
                with col_cl2:
                    if st.button("🧹 Clear All Stored Evidence for this Profile", key="btn_clear_ledger"):
                        DB.clear_candidate_evidence(st.session_state.candidate_id)
                        recalculate_profile(st.session_state.candidate_id)
                        st.success("✓ Evidence ledger cleared successfully.")
                        st.rerun()
            else:
                st.info("No evidence registered yet.")
        else:
            st.info("Enter your email above to inspect your evidence ledger.")

    st.markdown("---")
    col_p1, col_p2 = st.columns([5, 5])
    with col_p1:
        fresh_audit = st.checkbox(
            "🧹 Fresh Profile Audit (evaluate only newly entered inputs)",
            value=True,
            key="step1_fresh_audit",
            help="When checked, any historical evidence cached for this email address is cleared before ingesting, ensuring capability scores strictly match only the handles and files you provided right now."
        )
    with col_p2:
        if st.button("🚀 Ingest Evidence & Proceed to Step 2: Capability Profile →", type="primary", use_container_width=True, key="btn_next_step1_main"):
            clean_email = email_val.strip().lower()
            if not clean_email or "@" not in clean_email:
                st.error("Please enter a valid email address in Section 1 before proceeding.")
            else:
                st.session_state.user_email = clean_email
                st.session_state.user_name = name_val.strip()
                st.session_state.user_exp = exp_val

                cand = DB.get_or_create_candidate(clean_email, name=st.session_state.user_name or "Data Practitioner", title="Data Practitioner", years_exp=exp_val)
                st.session_state.candidate_id = cand["candidate_id"]
                st.session_state.candidate_info = cand

                # If Fresh Audit is selected, purge prior historical sources for clean evaluation
                if fresh_audit:
                    DB.clear_candidate_evidence(cand["candidate_id"])
                else:
                    # Replace per-source evidence if newly provided so duplicate sources don't accumulate
                    target_gh_chk = gh_input_user.strip()
                    target_lc_chk = lc_input_user.strip()
                    if target_gh_chk:
                        DB.remove_source(cand["candidate_id"], "GITHUB")
                    if target_lc_chk:
                        DB.remove_source(cand["candidate_id"], "SKILL_PLATFORM")
                    if uploaded_resume is not None:
                        DB.remove_source(cand["candidate_id"], "RESUME")

                feedback = []

                # GitHub Fetch if provided
                target_gh = gh_input_user.strip()
                if target_gh:
                    with st.spinner(f"Auditing GitHub @{target_gh}..."):
                        gh_conn = GitHubConnector()
                        evi_gh = gh_conn.fetch_public_profile(target_gh)
                        src_id = DB.register_source(cand["candidate_id"], "GITHUB", f"github.com/{target_gh}")
                        DB.add_evidence(
                            cand["candidate_id"], src_id, evi_gh["evidence_type"],
                            evi_gh["title"], evi_gh["description"], evi_gh["verification_strength"],
                            evi_gh["base_score"], evi_gh["recency_months"], evi_gh["skill_events"]
                        )
                        event_cnt = len(evi_gh["skill_events"])
                        if event_cnt > 0:
                            feedback.append(f"✓ GitHub @{target_gh}: Audited public repos -> {event_cnt} verified skill signals.")
                        else:
                            feedback.append(f"ℹ️ GitHub @{target_gh}: 0 public repos found.")

                # LeetCode Fetch if provided
                target_lc = lc_input_user.strip()
                if target_lc:
                    with st.spinner(f"Querying LeetCode for @{target_lc}..."):
                        lc_conn = LeetCodeConnector()
                        evi_lc = lc_conn.fetch_public_stats(target_lc)
                        raw = evi_lc.get("raw_counts", {})
                        if raw.get("All", 0) > 0:
                            src_id = DB.register_source(cand["candidate_id"], "SKILL_PLATFORM", f"leetcode.com/{target_lc}")
                            DB.add_evidence(
                                cand["candidate_id"], src_id, evi_lc["evidence_type"],
                                evi_lc["title"], evi_lc["description"], evi_lc["verification_strength"],
                                evi_lc["base_score"], evi_lc["recency_months"], evi_lc["skill_events"]
                            )
                            feedback.append(f"✓ LeetCode @{target_lc}: {raw.get('All', 0)} accepted problems ({raw.get('Hard', 0)} Hard).")
                        else:
                            feedback.append(f"ℹ️ LeetCode @{target_lc}: 0 accepted submissions found.")

                # Resume Processing if uploaded
                if uploaded_resume is not None:
                    with st.spinner(f"Extracting skills from resume {uploaded_resume.name}..."):
                        r_conn = ResumeConnector()
                        resume_bytes = uploaded_resume.getvalue() if hasattr(uploaded_resume, "getvalue") else uploaded_resume.read()
                        evi = r_conn.process(filename=uploaded_resume.name, file_bytes=resume_bytes)
                        src_id = DB.register_source(cand["candidate_id"], "RESUME", uploaded_resume.name)
                        DB.add_evidence(
                            cand["candidate_id"], src_id, evi["evidence_type"],
                            evi["title"], evi["description"], evi["verification_strength"],
                            evi["base_score"], evi["recency_months"], evi["skill_events"]
                        )
                        feedback.append(f"✓ Resume: Extracted verified skills from {uploaded_resume.name}.")

                recalculate_profile(cand["candidate_id"])
                if feedback:
                    st.session_state.fetch_feedback = feedback
                st.session_state.current_step = 2
                st.rerun()


# =====================================================================
# STEP 2: CAPABILITY PROFILE
# =====================================================================
elif st.session_state.current_step == 2:
    st.markdown("""
    <div class="step-header-box">
        <div class="step-header-title">Step 2: Verified Capability Profile</div>
        <div class="step-header-desc">Derived from authentic evidence artifacts with mathematical skill half-life decay. Every score contains an auditable 'WHY' provenance explanation.</div>
    </div>
    """, unsafe_allow_html=True)

    caps = st.session_state.derived_capabilities
    if not caps and st.session_state.candidate_id:
        recalculate_profile(st.session_state.candidate_id)
        caps = st.session_state.derived_capabilities

    if not caps:
        st.warning("⚠️ No capability profile generated yet. Please enter your email or connect a source in Step 1.")
        if st.button("👈 Return to Step 1: Identity & Evidence"):
            st.session_state.current_step = 1
            st.rerun()
    else:
        col_c1, col_c2 = st.columns(2)
        keys_list = list(SKILL_DISPLAY_NAMES.keys())
        half = len(keys_list) // 2
        conf_map = st.session_state.confidence_scores
        reasons_map = st.session_state.provenance_reasons

        for idx, s_key in enumerate(keys_list):
            score = caps.get(s_key, 0.20)
            score_pct = int(round(score * 100))
            conf = conf_map.get(s_key, 0.70)
            conf_label = "Strong Evidence" if conf >= 0.80 else ("Moderate Evidence" if conf >= 0.60 else "Developing")
            reasons = reasons_map.get(s_key, ["Baseline entry level."])
            disp_name = SKILL_DISPLAY_NAMES.get(s_key, s_key.title())

            target_col = col_c1 if idx < half else col_c2
            with target_col:
                st.markdown(f"""
                <div class="provenance-card">
                    <div class="provenance-header">
                        <span class="provenance-title">{disp_name}</span>
                        <span style="font-weight: 700; font-size: 1.15rem; color: #0284c7;">{score_pct}%</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 6px;">
                        Confidence: <strong>{conf_label} ({conf*100:.0f}%)</strong>
                    </div>
                    <div class="opp-fit-bar-bg">
                        <div class="opp-fit-bar-fill" style="width: {score_pct}%;"></div>
                    </div>
                    <div class="why-box">
                        <strong>WHY THIS SCORE?</strong>
                        <ul>
                            {"".join(f"<li>{r}</li>" for r in reasons)}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.plotly_chart(
            plot_capability_radar(caps, title=f"Capability Radar — {st.session_state.user_name or 'Candidate'}"),
            width="stretch"
        )

        with st.expander("🔬 Evidence Confidence & 18-Month Skill Half-Life Audit (Deck Slide 04)"):
            st.markdown("""
            #### How NEXUS Quantifies Capability Defensibility
            Unlike conventional job platforms that treat all self-declared resume bullets equally, **NEXUS computes capability confidence through source verification tiers, recency decay, and multi-source cross-corroboration**.
            """)

            col_cf1, col_cf2 = st.columns(2)
            with col_cf1:
                st.markdown("""
                ##### 1. Evidence Verification Hierarchy
                | Evidence Tier | Verification Weight | Verified Artifact Examples |
                |---|---|---|
                | **STRONG (1.00)** | Full Verification | Production GitHub repos, verified client deliverables |
                | **STRONG (1.00)** | Full Verification | Real-world competition pipelines, accepted PRs |
                | **MEDIUM (0.85)** | High Credibility | Accredited university coursework, graded lab exams |
                | **SUPPORTING (0.75)** | Moderate Signal | Industry certifications, verified MOOC credentials |
                | **WEAK / SELF (0.50)** | Baseline Entry | Self-declared profile claims & informal interest |
                """)

            with col_cf2:
                st.markdown("""
                ##### 2. Mathematical Skill Half-Life Decay (18-Month)
                Without ongoing active demonstration, technical skill confidence decays exponentially over an 18-month half-life:
                $$Confidence(e) = BaseScore \\times Weight_{strength} \\times 2^{-\\frac{\\Delta t}{18\\text{ months}}}$$
                
                * **Recency Guarantee:** Skills demonstrated within the last 30 days retain ~98%+ confidence.
                * **Stale Evidence Protection:** Inactive skills older than 36 months naturally decay to baseline, protecting against obsolete candidate claims.
                * **Multi-Source Corroboration:** When 2+ independent sources (e.g. GitHub + Coursework) validate the same capability, a **+10% cross-validation bonus** is awarded.
                """)

        st.markdown("---")
        col_b1, col_b2 = st.columns([3, 3])
        with col_b1:
            if st.button("← Back to Step 1: Evidence", use_container_width=True, key="btn_prev_step2"):
                st.session_state.current_step = 1
                st.rerun()
        with col_b2:
            if st.button("Proceed to Step 3: Market Fit →", type="primary", use_container_width=True, key="btn_next_step2"):
                st.session_state.current_step = 3
                st.rerun()


# =====================================================================
# STEP 3: MARKET FIT & POSITIONS
# =====================================================================
elif st.session_state.current_step == 3:
    st.markdown("""
    <div class="step-header-box">
        <div class="step-header-title">Step 3: Market Opportunity Alignment</div>
        <div class="step-header-desc">Evaluating your verified capabilities against 15,841 real job postings across India's top technology hubs.</div>
    </div>
    """, unsafe_allow_html=True)

    caps = st.session_state.derived_capabilities          # display keys (Step 2 only)
    exp = float(st.session_state.user_exp)
    effective_exp = exp                                   # Exact user experience (0.0 for freshers)

    if not caps and st.session_state.candidate_id:
        recalculate_profile(st.session_state.candidate_id)
        caps = st.session_state.derived_capabilities

    # Engine caps use role_profile keys — required for OpportunityEngine
    engine_caps = st.session_state.engine_capabilities or translate_to_engine_caps(caps)

    if not engine_caps:
        st.warning("⚠️ Please build or load an evidence profile in Step 1 first.")
        if st.button("👈 Go to Step 1"):
            st.session_state.current_step = 1
            st.rerun()
    else:
        with st.spinner("Evaluating multi-dimensional role compatibility..."):
            landscape = SYSTEM["opp_engine"].evaluate_candidate_landscape(engine_caps, effective_exp)

        df_roles = landscape["roles_table"]
        reachable_cnt = landscape["reachable_count"]
        stretch_cnt = landscape["stretch_count"]
        avg_fit = df_roles["compatibility_pct"].mean()

        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            st.markdown(render_kpi("Reachable Roles", f"{reachable_cnt}", "Fit >= 65% — Ready to interview"), unsafe_allow_html=True)
        with col_k2:
            st.markdown(render_kpi("Stretch Roles", f"{stretch_cnt}", "Fit 50-64% — 1-2 skills away"), unsafe_allow_html=True)
        with col_k3:
            st.markdown(render_kpi("Average Market Fit", f"{avg_fit:.1f}%", "Across all 10 analyzed role families"), unsafe_allow_html=True)

        st.markdown("#### Top Opportunities for Your Profile")
        for rank, (_, row) in enumerate(df_roles.head(3).iterrows(), 1):
            fit_pct = int(round(row["compatibility_pct"]))
            status = row["status"]
            badge_class = f"badge-{status.lower()}"
            status_label = "Strong Fit" if status == "Reachable" else ("Good Stretch" if status == "Stretch" else "Needs Work")
            role_name = row["role_name"]

            reqs = SYSTEM["role_profiles"].get(role_name, {})
            # role_profiles use engine_cap keys (sql_database, math_statistics etc.) — use engine_caps for lookup
            eng_display = {k: k.replace('_', ' ').title() for k in engine_caps}
            strengths = [eng_display.get(k, k.replace('_', ' ').title()) for k, w in reqs.items() if w >= 0.25 and engine_caps.get(k, 0.0) >= 0.55]
            developing = [eng_display.get(k, k.replace('_', ' ').title()) for k, w in reqs.items() if w >= 0.25 and 0.35 <= engine_caps.get(k, 0.0) < 0.55]
            missing = [eng_display.get(k, k.replace('_', ' ').title()) for k, w in reqs.items() if w >= 0.25 and engine_caps.get(k, 0.0) < 0.35]

            if strengths:
                strengths_label = ", ".join(strengths)
            elif developing:
                strengths_label = "Developing: " + ", ".join(developing)
            else:
                strengths_label = "⚠️ No verified competencies meeting market threshold yet"

            if missing:
                missing_label = ", ".join(missing)
            elif not strengths and not developing:
                missing_label = "All core domain competencies require evidence"
            else:
                missing_label = "No major gaps"

            st.markdown(f"""
            <div class="opp-card">
                <div class="opp-header">
                    <div>
                        <span style="font-size: 0.85rem; font-weight: 700; color: #64748b;">#{rank} OPPORTUNITY</span>
                        <h3 class="opp-title">{role_name}</h3>
                    </div>
                    <div><span class="{badge_class}">{status_label} ({fit_pct}%)</span></div>
                </div>
                <div class="opp-meta">
                    Benchmark Compensation: <strong>₹{row['average_salary_lakhs']:.1f}L</strong> &nbsp;|&nbsp;
                    Min Experience: <strong>{row['required_experience']:.1f} yrs</strong> &nbsp;|&nbsp;
                    Vacancies: <strong>{int(row['market_vacancies']):,}</strong>
                </div>
                <div class="opp-fit-bar-bg"><div class="opp-fit-bar-fill" style="width: {fit_pct}%;"></div></div>
                <div class="opp-skills-row">
                    <div class="opp-skills-col">
                        <div class="opp-skills-label strengths">✓ What Fits Well:</div>
                        <div>{strengths_label}</div>
                    </div>
                    <div class="opp-skills-col">
                        <div class="opp-skills-label gaps">△ Key Skill Gaps:</div>
                        <div>{missing_label}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📋 View All 10 Role Families Across the Market"):
            st.dataframe(df_roles[["role_name", "status", "compatibility_pct", "average_salary_lakhs", "market_vacancies"]].rename(columns={
                "role_name": "Role Family", "status": "Readiness", "compatibility_pct": "Fit Score (%)",
                "average_salary_lakhs": "Benchmark Salary (Lakhs)", "market_vacancies": "Market Vacancies"
            }), width="stretch")

        with st.expander("🤖 Empirical Machine Learning Signals (JDS & SDS Datasets)"):
            jds_lr = SYSTEM["jds_insights"]["logistic_regression"]
            sds_lr = SYSTEM["sds_insights"]["logistic_regression"]
            col_jm, col_sm = st.columns(2)
            with col_jm:
                st.markdown("**Junior Data Scientist: Technical Skills vs. Salary Hike**")
                st.caption(f"L2-Logistic Regression (ROC-AUC {jds_lr['roc_auc']:.3f}, 5-Fold CV {jds_lr['cv_accuracy_mean']*100:.1f}%)")
                st.plotly_chart(plot_coefficients_bar(jds_lr["coefficients"], "JDS: Technical Coefficients on Hike"), width="stretch")
            with col_sm:
                st.markdown("**Senior Data Scientist: Personality vs. Success**")
                st.caption(f"L2-Logistic Regression (ROC-AUC {sds_lr['roc_auc']:.3f}, 5-Fold CV {sds_lr['cv_accuracy_mean']*100:.1f}%)")
                st.plotly_chart(plot_coefficients_bar(sds_lr["coefficients"], "SDS: Trait Coefficients on Success"), width="stretch")

        st.markdown("---")
        col_b1, col_b2 = st.columns([3, 3])
        with col_b1:
            if st.button("← Back to Step 2: Capabilities", use_container_width=True, key="btn_prev_step3"):
                st.session_state.current_step = 2
                st.rerun()
        with col_b2:
            if st.button("Proceed to Step 4: Bottleneck Gap →", type="primary", use_container_width=True, key="btn_next_step3"):
                st.session_state.current_step = 4
                st.rerun()


# =====================================================================
# STEP 4: BOTTLENECK GAP
# =====================================================================
elif st.session_state.current_step == 4:
    st.markdown("""
    <div class="step-header-box">
        <div class="step-header-title">Step 4: Transition Bottleneck Analysis</div>
        <div class="step-header-desc">Pinpointing the single capability holding you back the most across all blocked roles in the market.</div>
    </div>
    """, unsafe_allow_html=True)

    caps = st.session_state.derived_capabilities
    exp = float(st.session_state.user_exp)
    effective_exp = exp

    if not caps and st.session_state.candidate_id:
        recalculate_profile(st.session_state.candidate_id)
        caps = st.session_state.derived_capabilities

    engine_caps = st.session_state.engine_capabilities or translate_to_engine_caps(caps)

    if not engine_caps:
        st.warning("⚠️ Please build or load an evidence profile in Step 1 first.")
        if st.button("👈 Go to Step 1"):
            st.session_state.current_step = 1
            st.rerun()
    else:
        with st.spinner("Calculating transition leverage scores..."):
            bottlenecks_df = SYSTEM["opp_engine"].detect_bottlenecks(engine_caps, effective_exp)

        if not bottlenecks_df.empty:
            top_bn = bottlenecks_df.iloc[0]
            top_skill_name = top_bn["display_name"]
            top_skill_key = top_bn["skill_key"]
            cand_curr = engine_caps.get(top_skill_key, 0.20)

            st.markdown(f"""
            <div class="hero-gap-box">
                <div class="hero-gap-tag">PRIMARY CAREER BOTTLENECK</div>
                <div class="hero-gap-title">{top_skill_name}</div>
                <div class="hero-gap-desc">
                    Your current verified score in this skill is <strong>{cand_curr*100:.0f}%</strong>.
                    Overcoming this specific deficit produces the highest simulated opportunity expansion across blocked roles.
                </div>
                <div class="hero-gap-bullet">✓ <strong>Substantial deficit</strong> between your verified score and market thresholds.</div>
                <div class="hero-gap-bullet">✓ <strong>High market demand</strong> across multiple high-paying analytics roles.</div>
                <div class="hero-gap-bullet">✓ <strong>Maximal unlock impact:</strong> multiple roles are blocked solely by this capability.</div>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(plot_bottlenecks_bar(bottlenecks_df), width="stretch")

            with st.expander("🔍 Mathematical Bottleneck Scoring Breakdown"):
                st.markdown("""
                $$Bottleneck = 0.35 \\times Gap + 0.35 \\times DemandWeight + 0.30 \\times ImpactScore$$
                """)
                st.dataframe(bottlenecks_df[["display_name", "priority_score", "average_gap_in_target_roles", "market_demand_weight", "unlocked_roles_count"]].rename(columns={
                    "display_name": "Capability", "priority_score": "Leverage Score", "average_gap_in_target_roles": "Average Gap",
                    "market_demand_weight": "Market Demand Weight", "unlocked_roles_count": "Unlocked Roles Count"
                }), width="stretch")
        else:
            st.success("Your profile is well-balanced across all target role requirements!")

        st.markdown("---")
        col_b1, col_b2 = st.columns([3, 3])
        with col_b1:
            if st.button("← Back to Step 3: Market Fit", use_container_width=True, key="btn_prev_step4"):
                st.session_state.current_step = 3
                st.rerun()
        with col_b2:
            if st.button("Proceed to Step 5: What-If Simulation →", type="primary", use_container_width=True, key="btn_next_step4"):
                st.session_state.current_step = 5
                st.rerun()


# =====================================================================
# STEP 5: WHAT-IF SIMULATION
# =====================================================================
elif st.session_state.current_step == 5:
    st.markdown("""
    <div class="step-header-box">
        <div class="step-header-title">Step 5: Counterfactual What-If Simulation</div>
        <div class="step-header-desc">Evaluating three targeted interventions to simulate your altered career landscape and compensation growth.</div>
    </div>
    """, unsafe_allow_html=True)

    caps = st.session_state.derived_capabilities
    exp = float(st.session_state.user_exp)
    effective_exp = exp

    if not caps and st.session_state.candidate_id:
        recalculate_profile(st.session_state.candidate_id)
        caps = st.session_state.derived_capabilities

    engine_caps = st.session_state.engine_capabilities or translate_to_engine_caps(caps)

    if not engine_caps:
        st.warning("⚠️ Please build or load an evidence profile in Step 1 first.")
        if st.button("👈 Go to Step 1"):
            st.session_state.current_step = 1
            st.rerun()
    else:
        with st.spinner("Simulating counterfactual interventions..."):
            all_sim_results = SYSTEM["simulator"].simulate_all_interventions(engine_caps, effective_exp)

        today_landscape = SYSTEM["opp_engine"].evaluate_candidate_landscape(engine_caps, effective_exp)
        today_reachable = today_landscape["reachable_count"]
        today_avg_sal = today_landscape["average_reachable_salary"]

        col_t0, col_t1, col_t2, col_t3 = st.columns(4)
        with col_t0:
            st.markdown(f"""
            <div class="kpi-container" style="border-top: 4px solid #64748b;">
                <div class="kpi-title">TODAY</div>
                <div class="kpi-value">{today_reachable}</div>
                <div class="kpi-subtext">Reachable roles today</div>
                <div style="font-size: 0.8rem; color: #475569; margin-top: 8px;">Avg Pay: <strong>₹{today_avg_sal:.1f}L</strong></div>
            </div>
            """, unsafe_allow_html=True)

        sim_map = {s["intervention"].get("preset_code", ""): s for s in all_sim_results}
        for col, code, label in [(col_t1, "INTERVENTION A", "PATH A: AI / ML"),
                                  (col_t2, "INTERVENTION B", "PATH B: Storytelling"),
                                  (col_t3, "INTERVENTION C", "PATH C: Data Engineering")]:
            s = sim_map.get(code)
            if s:
                gain = len(s["newly_unlocked_roles"])
                kpi_val = s['future_reachable_count'] if s['future_reachable_count'] > 0 else len(s['newly_unlocked_roles'])
                with col:
                    st.markdown(f"""
                    <div class="kpi-container" style="border-top: 4px solid #0284c7;">
                        <div class="kpi-title">{label}</div>
                        <div class="kpi-value">{kpi_val}</div>
                        <div class="kpi-subtext">+{gain} new role{'s' if gain != 1 else ''} (+{s['opportunity_expansion_pct']:.0f}%)</div>
                        <div style="font-size: 0.8rem; color: #0284c7; margin-top: 8px;">Salary Lift: <strong>+₹{s['salary_growth_lakhs']:.2f}L</strong></div>
                    </div>
                    """, unsafe_allow_html=True)

        # Highlight optimal pathway
        best_sim = max(all_sim_results, key=lambda x: (x["opportunity_expansion_pct"], x["salary_growth_lakhs"], x["vacancy_growth"]))
        best_inv = best_sim["intervention"]
        best_name = best_inv.get("short_name", best_inv.get("name", "Winning Path"))
        newly_unlocked = best_sim["newly_unlocked_roles"]
        vacancies_unlocked = best_sim.get("vacancy_growth", 0)

        st.markdown(f"""
        <div class="futures-winner-box">
            <div class="futures-winner-header"><span>⭐</span> HIGHEST-LEVERAGE GROWTH PATHWAY</div>
            <div class="futures-winner-title">{best_name}</div>
            <div style="font-size: 1rem; color: #166534; line-height: 1.5;">
                {best_inv.get('rationale', 'This pathway provides the highest transition leverage across the analysed market data.')}
            </div>
            <div class="futures-winner-metrics">
                <div class="futures-stat">
                    <div class="futures-stat-val">+{len(newly_unlocked)}</div>
                    <div class="futures-stat-label">Newly Reachable Roles<br>({', '.join(newly_unlocked) if newly_unlocked else 'Consolidation'})</div>
                </div>
                <div class="futures-stat">
                    <div class="futures-stat-val">+{vacancies_unlocked:,}</div>
                    <div class="futures-stat-label">Additional Open Positions</div>
                </div>
                <div class="futures-stat">
                    <div class="futures-stat-val">+₹{best_sim['salary_growth_lakhs']:.2f}L</div>
                    <div class="futures-stat-label">Average Pay Lift</div>
                </div>
                <div class="futures-stat">
                    <div class="futures-stat-val">+{best_sim['opportunity_expansion_pct']:.1f}%</div>
                    <div class="futures-stat-label">Total Opportunity Expansion</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(plot_three_futures_comparison(all_sim_results), width="stretch")

        with st.expander("📊 3 Simulated Futures Comparative Matrix (The Counterfactual Layer — Slide 09)"):
            st.markdown("""
            #### Side-by-Side Transition Comparison Across All 3 Pathways
            $$\\text{Opportunity Expansion}(I) = \\frac{|\\text{Roles after intervention}| - |\\text{Roles before}|}{|\\text{Roles before}|}$$
            """)
            matrix_data = []
            for s in all_sim_results:
                inv = s["intervention"]
                code = inv.get("preset_code", "")
                p_name = inv.get("short_name", inv.get("name", code))
                g_roles = s["future_reachable_count"] - s["baseline_reachable_count"]
                unlocked_str = ", ".join(s["newly_unlocked_roles"]) if s["newly_unlocked_roles"] else "Consolidation"

                fut_caps = s["future_capabilities"]
                fut_bns = SYSTEM["opp_engine"].detect_bottlenecks(fut_caps, effective_exp)
                rem_bn = fut_bns.iloc[0]["display_name"] if not fut_bns.empty else "Balanced"

                matrix_data.append({
                    "Intervention Pathway": f"{code}: {p_name}",
                    "Targeted Core Skills": ", ".join(k.replace('_', ' ').title() for k in inv.get("skills_affected", {}).keys()),
                    "Reachable Roles": f"{s['baseline_reachable_count']} → {s['future_reachable_count']} (+{g_roles})",
                    "Opportunity Expansion": f"+{s['opportunity_expansion_pct']:.1f}%",
                    "Newly Unlocked Roles": unlocked_str,
                    "Projected Avg Pay": f"₹{s['future_avg_salary']:.2f}L (+₹{s['salary_growth_lakhs']:.2f}L)",
                    "Remaining Bottleneck": rem_bn
                })

            st.dataframe(pd.DataFrame(matrix_data), width="stretch")

        # ----------------------------------------------------
        # Expected Offer in Expected Role Calculator
        # ----------------------------------------------------
        st.markdown("---")
        st.markdown("### 💼 Expected Offer in Your Target Role")
        st.caption("Forecast your anticipated compensation package based on empirical market salary bands and your quantified capability qualification score.")

        available_roles = list(today_landscape["roles_table"]["role_name"])
        default_role_idx = 0
        if newly_unlocked:
            for i, r in enumerate(available_roles):
                if r in newly_unlocked:
                    default_role_idx = i
                    break

        col_sel_role, col_sel_inv = st.columns([6, 4])
        with col_sel_role:
            target_role_chosen = st.selectbox(
                "🎯 Select Your Expected / Target Role:",
                options=available_roles,
                index=default_role_idx,
                key="step5_target_role_select"
            )

        with col_sel_inv:
            inv_options = [s["intervention"].get("preset_code", "") for s in all_sim_results]
            best_code = best_inv.get("preset_code", "INTERVENTION A")
            inv_idx = inv_options.index(best_code) if best_code in inv_options else 0
            selected_inv_code = st.selectbox(
                "🔮 Evaluate Post-Learning Offer Under:",
                options=inv_options,
                format_func=lambda c: f"{c}: {sim_map.get(c, {}).get('intervention', {}).get('short_name', c)}",
                index=inv_idx,
                key="step5_inv_select"
            )

        # Retrieve simulation data for selected intervention
        chosen_sim = sim_map.get(selected_inv_code, best_sim)
        chosen_future_roles_df = chosen_sim["roles_comparison_table"]

        # Calculate today's offer metrics
        t_row = today_landscape["roles_table"][today_landscape["roles_table"]["role_name"] == target_role_chosen].iloc[0]
        today_compat_pct = float(t_row["compatibility_pct"])
        today_offer = compute_expected_offer(target_role_chosen, today_compat_pct, effective_exp)

        # Calculate future offer metrics
        f_row = chosen_future_roles_df[chosen_future_roles_df["role_name"] == target_role_chosen].iloc[0]
        future_compat_pct = float(f_row["score_after"])
        future_offer = compute_expected_offer(target_role_chosen, future_compat_pct, effective_exp + 0.5)

        offer_lift = round(future_offer["expected_salary"] - today_offer["expected_salary"], 2)
        offer_lift_pct = round((offer_lift / max(today_offer["expected_salary"], 0.1)) * 100.0, 1)

        # Render Expected Offer Hero Box
        today_badge = "badge-reachable" if today_offer["status"] == "Reachable" else ("badge-stretch" if today_offer["status"] == "Stretch" else "badge-blocked")
        future_badge = "badge-reachable" if future_offer["status"] == "Reachable" else ("badge-stretch" if future_offer["status"] == "Stretch" else "badge-blocked")

        st.markdown(f"""
        <div class="offer-box">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px;">
                <div>
                    <span style="font-size: 0.8rem; font-weight: 800; color: #166534; letter-spacing: 0.05em; text-transform: uppercase;">EMPIRICAL COMPENSATION FORECAST</span>
                    <h3 style="margin: 2px 0 0 0; font-size: 1.6rem; font-weight: 900; color: #064e3b;">{target_role_chosen}</h3>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.82rem; color: #166534; font-weight: 700;">Market Vacancies:</span>
                    <strong style="color: #064e3b; font-size: 1rem;">{today_offer['vacancies']:,} Openings</strong>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 16px;">
                <div style="background: white; border-radius: 8px; padding: 14px; border: 1px solid #bbf7d0;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #64748b;">CURRENT OFFER POTENTIAL</div>
                    <div style="font-size: 1.7rem; font-weight: 900; color: #334155;">₹{today_offer['expected_salary']:.2f}L</div>
                    <div style="font-size: 0.78rem; margin-top: 4px;">Fit: <strong>{today_compat_pct:.1f}%</strong> &nbsp;<span class="{today_badge}">{today_offer['status']}</span></div>
                </div>
                <div style="background: white; border-radius: 8px; padding: 14px; border: 2px solid #22c55e;">
                    <div style="font-size: 0.78rem; font-weight: 800; color: #166534;">PROJECTED EXPECTED OFFER</div>
                    <div class="offer-val-lg">₹{future_offer['expected_salary']:.2f}L</div>
                    <div style="font-size: 0.78rem; margin-top: 4px;">Fit: <strong>{future_compat_pct:.1f}%</strong> &nbsp;<span class="{future_badge}">{future_offer['status']}</span></div>
                </div>
                <div style="background: white; border-radius: 8px; padding: 14px; border: 1px solid #bbf7d0;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #64748b;">ESTIMATED INCREMENT</div>
                    <div style="font-size: 1.7rem; font-weight: 900; color: #0284c7;">+₹{offer_lift:.2f}L</div>
                    <div style="font-size: 0.78rem; color: #0369a1; margin-top: 4px; font-weight: 700;">+{offer_lift_pct:.1f}% salary growth</div>
                </div>
                <div style="background: white; border-radius: 8px; padding: 14px; border: 1px solid #bbf7d0;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #64748b;">MARKET BENCHMARK RANGE</div>
                    <div style="font-size: 1rem; font-weight: 800; color: #0f172a; margin: 4px 0;">₹{today_offer['min_salary']:.1f}L — ₹{today_offer['max_salary']:.1f}L</div>
                    <div style="font-size: 0.76rem; color: #64748b;">Market Median: <strong>₹{today_offer['avg_salary']:.1f}L</strong></div>
                </div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.7); border-radius: 6px; padding: 10px 14px; font-size: 0.82rem; color: #14532d;">
                <strong>Estimated Package Structure:</strong> Fixed Base (~₹{future_offer['expected_salary']*0.80:.2f}L / 80%) &nbsp;•&nbsp; Performance Bonus (~₹{future_offer['expected_salary']*0.15:.2f}L / 15%) &nbsp;•&nbsp; Retention/Benefits (~₹{future_offer['expected_salary']*0.05:.2f}L / 5%)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ----------------------------------------------------
        # Targeted Learning Curriculum: What You Need to Learn
        # ----------------------------------------------------
        st.markdown("---")
        st.markdown("### 📚 Targeted Learning Curriculum: What You Need to Learn")
        st.caption("Actionable technical roadmaps with core modules, tools, and capstone milestones to bridge your deficits and qualify for target offers.")

        tab_curr_win, tab_curr_a, tab_curr_b, tab_curr_c = st.tabs([
            f"⭐ Recommended Pathway ({best_name})",
            "Path A: AI & Machine Learning",
            "Path B: Analytics & Storytelling",
            "Path C: Data Engineering & MLOps"
        ])

        def _render_curriculum_view(code_key: str):
            curr_info = INTERVENTION_CURRICULUM.get(code_key, {})
            if not curr_info:
                st.info("Curriculum details unavailable.")
                return

            st.markdown(f"#### {curr_info['title']}")
            st.markdown(f"**Track:** `{curr_info['category']}` &nbsp;|&nbsp; **Estimated Commitment:** `{curr_info['duration']}`")

            # Skills Affected
            st.markdown("##### 📈 Targeted Capability Upgrades:")
            col_sk = st.columns(len(curr_info["skills_boosted"]))
            for col_s, (s_key, s_label, s_boost) in zip(col_sk, curr_info["skills_boosted"]):
                cur_lvl = engine_caps.get(s_key, 0.20)
                tar_lvl = min(1.0, cur_lvl + float(s_boost.replace('+', '').replace('%', '')) / 100.0)
                with col_s:
                    st.markdown(f"""
                    <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px; text-align: center;">
                        <div style="font-size: 0.78rem; font-weight: 700; color: #475569;">{s_label}</div>
                        <div style="font-size: 1.15rem; font-weight: 900; color: #0284c7; margin: 4px 0;">{cur_lvl*100:.0f}% → {tar_lvl*100:.0f}%</div>
                        <span class="status-pill info">{s_boost} boost</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("##### 📖 Curriculum Modules & Core Concepts to Learn:")
            for mod in curr_info["modules"]:
                st.markdown(f"""
                <div class="curriculum-card">
                    <div class="curriculum-module-title">Module {mod['num']}: {mod['name']}</div>
                    <div style="font-size: 0.86rem; color: #334155; line-height: 1.5; margin-top: 4px;">
                        <strong>Key Concepts to Master:</strong> {mod['topics']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            col_tl, col_cp = st.columns([4, 6])
            with col_tl:
                st.markdown("##### 🛠️ Core Tools & Frameworks:")
                tools_html = "".join(f"<span class='curriculum-badge'>{t}</span>" for t in curr_info["key_tools"])
                st.markdown(f"<div style='margin-top: 6px;'>{tools_html}</div>", unsafe_allow_html=True)
            with col_cp:
                st.markdown("##### 🏆 Portfolio Capstone to Build:")
                st.markdown(f"""
                <div style="background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 10px 12px; font-size: 0.84rem; color: #166534; line-height: 1.45;">
                    {curr_info['capstone']}
                </div>
                """, unsafe_allow_html=True)

        with tab_curr_win:
            _render_curriculum_view(best_inv.get("preset_code", "INTERVENTION A"))
        with tab_curr_a:
            _render_curriculum_view("INTERVENTION A")
        with tab_curr_b:
            _render_curriculum_view("INTERVENTION B")
        with tab_curr_c:
            _render_curriculum_view("INTERVENTION C")

        st.markdown("---")
        col_b1, col_b2 = st.columns([3, 3])
        with col_b1:
            if st.button("← Back to Step 4: Bottlenecks", use_container_width=True, key="btn_prev_step5"):
                st.session_state.current_step = 4
                st.rerun()
        with col_b2:
            if st.button("Proceed to Step 6: Scientific Audit →", type="primary", use_container_width=True, key="btn_next_step5"):
                st.session_state.current_step = 6
                st.rerun()


# =====================================================================
# STEP 6: SCIENTIFIC AUDIT & METHODOLOGY
# =====================================================================
elif st.session_state.current_step == 6:
    st.markdown("""
    <div class="step-header-box">
        <div class="step-header-title">Step 6: Scientific Audit, Data Governance & Models</div>
        <div class="step-header-desc">Full technical audit trail documenting data hygiene, relational schemas, model evaluation metrics, and scoring formulas.</div>
    </div>
    """, unsafe_allow_html=True)

    tab_a1, tab_a2, tab_a3, tab_a4, tab_a5 = st.tabs([
        "📊 The 4 Hackathon Datasets",
        "🗄️ SQLite Candidate Database",
        "🤖 Predictive Model Validation",
        "📐 Mathematical Formulas",
        "🌐 Ecosystem Roadmap & Graph Architecture"
    ])

    with tab_a1:
        q_rep = SYSTEM["quality_reports"]
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"**Analytics Jobs Dataset:** {q_rep['analytics_jobs']['original_rows']:,} rows ({q_rep['analytics_jobs']['cleaned_rows']:,} cleaned) · Standardized 11 role families")
            st.markdown(f"**Junior Data Scientist Skills:** {q_rep['jds_skills']['original_rows']} observations · Technical skills vs. salary-hike outcome")
        with col_d2:
            st.markdown(f"**Data Science Jobs Dataset:** {q_rep['datascience_jobs']['original_rows']:,} records · {SYSTEM['headline_metrics']['total_market_vacancies']:,} total vacancies")
            st.markdown(f"**Senior Data Scientist Personality:** {q_rep['sds_personality']['original_rows']} observations · Personality traits vs. senior practitioner success")

    with tab_a2:
        if st.session_state.candidate_id:
            st.markdown(f"**Candidate:** `{st.session_state.user_email}` (`{st.session_state.candidate_id}`)")
            ev_summary = DB.get_candidate_evidence_summary(st.session_state.candidate_id)
            if ev_summary:
                st.dataframe(pd.DataFrame(ev_summary), width="stretch")
            st.markdown("#### Raw Skill Demonstration Events")
            ev_events = DB.get_skill_events_for_candidate(st.session_state.candidate_id)
            if ev_events:
                st.dataframe(pd.DataFrame(ev_events), width="stretch")
        else:
            st.info("No active candidate loaded. Go to Step 1 to load or identify a candidate.")

    with tab_a3:
        jds_lr = SYSTEM["jds_insights"]["logistic_regression"]
        sds_lr = SYSTEM["sds_insights"]["logistic_regression"]
        col_jm2, col_sm2 = st.columns(2)
        with col_jm2:
            st.markdown("**JDS Model (Junior Data Scientist)**")
            st.write(f"- Algorithm: L2-Regularized Logistic Regression")
            st.write(f"- 5-Fold CV Accuracy: {jds_lr['cv_accuracy_mean']*100:.1f}% (±{jds_lr['cv_accuracy_std']*100:.1f}%)")
            st.write(f"- Holdout ROC-AUC: {jds_lr['roc_auc']:.3f}")
            st.plotly_chart(plot_confusion_matrix_heatmap(jds_lr['confusion_matrix'], ["Normal", "High Hike"], "JDS Confusion Matrix"), width="stretch")
        with col_sm2:
            st.markdown("**SDS Model (Senior Data Scientist)**")
            st.write(f"- Algorithm: L2-Regularized Logistic Regression")
            st.write(f"- 5-Fold CV Accuracy: {sds_lr['cv_accuracy_mean']*100:.1f}% (±{sds_lr['cv_accuracy_std']*100:.1f}%)")
            st.write(f"- Holdout ROC-AUC: {sds_lr['roc_auc']:.3f}")
            st.plotly_chart(plot_confusion_matrix_heatmap(sds_lr['confusion_matrix'], ["Normal", "High Success"], "SDS Confusion Matrix"), width="stretch")

    with tab_a4:
        st.markdown("""
        #### 1. Multi-Dimensional Role Compatibility
        $$Compatibility(c, r) = 0.55 \\cdot Cov_{core}(c, r) + 0.25 \\cdot Cov_{sec}(c, r) + 0.20 \\cdot Cov_{exp}(c, r) - Pen_{deficit}(c, r)$$

        #### 2. Transition Bottleneck Detection
        $$BottleneckPriority(s) = 0.35 \\cdot Gap(s) + 0.35 \\cdot DemandWeight(s) + 0.30 \\cdot ImpactScore(s)$$

        #### 3. Evidence Skill Half-Life Decay (18-month)
        $$Confidence(e) = BaseScore \\times StrengthMultiplier \\times 2^{-\\frac{RecencyMonths}{18.0}}$$

        #### 4. Counterfactual Simulation Operator
        $$c' = \\tau(c, I) = \\max(c,\\ c + \\Delta_I)$$
        $$\\text{Opportunity Expansion} = \\frac{\\text{ReachableAfter} - \\text{ReachableBefore}}{\\max(\\text{ReachableBefore}, 1)}$$
        """)

    with tab_a5:
        st.markdown("""
        ### 🌐 Build for Bharat 2.0 • Multi-Stakeholder Ecosystem Architecture
        *Candidate-Only MVP Live Today · Recruiter, Organization, and Institutional Experiences on Active Roadmap (Deck Slides 02, 05, 07, 08)*
        """)

        col_ec1, col_ec2 = st.columns(2)
        with col_ec1:
            st.markdown("""
            #### 🏛️ The 4 Stakeholder Pillars (Slide 05)
            1. **👤 Candidate Experience (Live MVP Built First):**
               * Solves the hardest decision loop first: *Evidence → Capability State → Intervention → Counterfactual State → Opportunity Expansion*.
               * Allows engineering undergraduates and working practitioners to test *"If I learn X, what roles unlock?"*
            2. **🏢 Recruiter Portal (Roadmap):**
               * Replaces blind resume keyword matching with **verified evidence-backed transferable capability graphs**.
               * Audits candidate repos and verified coursework directly to eliminate credential fraud.
            3. **🏬 Organization Intelligence (Roadmap):**
               * Internal capability shortage forecasting.
               * **"Hire vs. Reskill" Cost/Time Optimization:** Models whether it is faster and cheaper to upskill existing analysts into data engineers or hire externally.
            4. **🎓 Academic Institutions & Colleges (Roadmap):**
               * **Curriculum ↔ Live Market Alignment:** Feeds live hiring signals back to university departments to modernize course syllabi.
            """)

        with col_ec2:
            st.markdown("""
            #### ⚙️ Graph-First Decision Layer & Defensibility (Slide 04 & 07)
            ```
            [Messy Candidate Evidence]
                 ↓  NLP / Entity Parsing
            [Person ↔ Evidence Graph Node]
                 ↓  Verification Weight & 18-Mo Decay
            [Quantified Capability State]
                 ↓  Counterfactual Operator τ(c, I)
            [Simulated Opportunity & Reachable Roles]
            ```

            * **Not an LLM Wrapper:** Deterministic mathematical operators, L2-regularized logistic regressions, and reproducible graph queries.
            * **Market Platform Benchmark:**
              * *LinkedIn / Naukri:* Discovery & keyword matching
              * *National Career Service (NCS):* Job listings & career services
              * *NEXUS:* Counterfactual transition simulation layer
            * **Data Governance Guarantee (Slide 06):**
              * 100% Consent-driven architecture.
              * Strict PII minimization — profiles remain non-identifiable until mutual recruiter outreach opt-in.
            """)

    st.markdown("---")
    col_b1, col_b2 = st.columns([3, 3])
    with col_b1:
        if st.button("← Back to Step 5: Simulations", use_container_width=True, key="btn_prev_step6"):
            st.session_state.current_step = 5
            st.rerun()
    with col_b2:
        if st.button("Return to Step 1: Identity & Evidence", use_container_width=True, key="btn_home_step6"):
            st.session_state.current_step = 1
            st.rerun()
