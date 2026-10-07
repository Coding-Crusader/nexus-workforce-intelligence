# NEXUS: Workforce Transition Intelligence
> *"Don't just measure where talent is. Model where talent can go."*

---

## 1. Executive Summary

**NEXUS** is an empirical workforce transition intelligence and counterfactual simulation platform built for the **SAS University Hackathon**. 

Conventional workforce tools suffer from two systemic failures:
1. **Static Role Matching:** They tell candidates what jobs they match *today*, reinforcing career inertia and trapping non-traditional talent in low-mobility tiers.
2. **Generic LLM Hallucinations:** They generate ungrounded advice using external black-box chatbots rather than calculating actual labor market demand, salary benchmarks, and measurable capability gaps.

NEXUS re-architects career mobility into a **deterministic counterfactual simulation**:
```
Evidence
  ↓
Capability State
  ↓
Opportunity / Role Mapping
  ↓
Targeted Intervention
  ↓
Counterfactual Future State
  ↓
Changed Opportunity Landscape
  ↓
Quantitative Explanation ("Why This?")
```

Instead of simply recommending a candidate's weakest skill, NEXUS evaluates:
$$\text{Transition Leverage} = \text{Candidate Gap} \times \text{Market Demand} \times \text{Opportunity Expansion Impact}$$

---

## 2. Hackathon Datasets & Empirical Ingestion

NEXUS runs **100% locally** on the actual competition datasets without synthetic substitutions:

| Dataset | Records / Rows | Scope & Description | Key Analytical Insights |
|---|---|---|---|
| **`Analytics Jobs.csv`** | **15,841** rows | Comprehensive job postings across India with raw unstructured `key_skills`, salary bands, and experience ranges. | 8,500+ raw skill mentions parsed into 10 canonical skill families. Top metropolitan hubs: Bengaluru (21.0%), Mumbai (12.6%), Gurgaon (8.3%). |
| **`DataScience Jobs.csv`** | **1,602** postings (**93,005** vacancies) | Enterprise hiring across 642 leading tech/consulting employers (TCS, Accenture, IBM, Cognizant, Infosys). | Standardized 10 canonical role families. Average market salary: **13.23L INR** (ranges from 5.71L for Data Analyst to 25.09L for Data Architect). |
| **`JDS Skill Traits.xlsx`** | **139** candidates | Junior Data Scientists measured across 5 technical skill dimensions (1-5 scale) against performance-based salary hike (high/low). | **Dashboard & storytelling skills** ($r=0.554$, $\beta=1.186$) and **maths/stats** ($r=0.524$, $\beta=1.451$) are the primary drivers of top-tier salary hikes. 5-Fold CV Accuracy: **84.1%**, ROC-AUC: **0.919**. |
| **`SDS Personality Traits.xlsx`** | **161** practitioners | Senior & customer-facing Data Scientists measured on the Big Five personality spectrum (OCEAN) against organizational success (high/low). | **Conscientiousness** ($r=0.680$, $+17.9$ pts) and **Openness to Experience** ($r=0.671$, $+15.2$ pts) show dominant associations. Neuroticism exhibits negligible correlation ($r=-0.006$). 5-Fold CV Accuracy: **91.3%**, ROC-AUC: **0.980**. |

---

## 3. System Architecture & Modularity

The application adheres to clean separation of concerns (**Data ≠ Business Logic ≠ User Interface**):

```
NEXUS/
│
├── app.py                     # Streamlit multi-page executive interface
├── requirements.txt           # Verified dependencies
├── README.md                  # Comprehensive documentation & methodology
│
├── backend/                   # Business logic and analytical engines
│   ├── __init__.py
│   ├── data_loader.py         # Multi-source ingestion & Streamlit caching
│   ├── data_cleaner.py        # Reproducible preprocessing & data quality auditing
│   ├── market_analysis.py     # Vacancy, salary, experience, and geo analytics
│   ├── skill_analysis.py      # RapidFuzz normalization & empirical role profiles
│   ├── success_analysis.py    # Interpretable ML (Logistic Regression & Decision Trees)
│   ├── opportunity_engine.py  # Multi-dimensional role compatibility & bottleneck scoring
│   ├── simulation.py          # Deterministic counterfactual engine & expansion metrics
│   ├── resume_parser.py       # 100% local, privacy-safe PDF/DOCX skill extractor
│   └── main.py                # Standalone pipeline CLI runner
│
├── frontend/                  # Presentation layer
│   ├── __init__.py
│   ├── styles.py              # Executive typography & CSS styles
│   └── ui.py                  # High-density Plotly charting components
│
├── config/                    # Externalized system configuration
│   ├── settings.json          # Scoring weights, thresholds, paths
│   ├── skill_aliases.json     # Normalization dictionaries for 10 skill families
│   └── interventions.json     # Empirically grounded upskilling interventions
│
├── data/
│   ├── raw/                   # Canonical datasets
│   ├── processed/             # Preprocessed data caches
│   └── demo_candidate.json    # Aarav Sharma baseline candidate profile
│
└── tests/
    └── test_nexus.py          # Pytest suite with 100% pass rate
```

---

## 4. Key Innovations

### A. The Counterfactual Transition Simulator
The core innovation of NEXUS is asking:
> *"What capability change matters most for career trajectory?"*

Given an active candidate profile and an intervention, NEXUS calculates:
$$\text{Opportunity Expansion} = \frac{\text{Reachable Roles}_{\text{after}} - \text{Reachable Roles}_{\text{before}}}{\text{Reachable Roles}_{\text{before}}}$$

For our demo candidate (**Aarav Sharma, Associate Data Analyst**):
- **Baseline State:** Qualified for entry Data Analyst & Business Analyst (8 reachable roles, 86,937 vacancies, 9.07L avg salary). Blocked from Senior tracks.
- **Intervention: Full-Stack MLOps & Production Deployment:**
  - Capability gains: Cloud/MLOps ($0.25 \to 0.60$), ML ($0.35 \to 0.55$), Big Data ($0.20 \to 0.40$).
  - **Opportunity Expansion:** **+25.0%** (8 $\to$ 10 reachable roles).
  - **Newly Reachable Roles:** **Senior Data Engineer** and **Senior Data Scientist**.
  - **Market Vacancy Expansion:** **+5,540** additional accessible positions.
  - **Average Salary Trajectory:** **9.07L $\to$ 11.39L (+2.32L INR gain)**.

### B. High-Leverage Bottleneck Detection
Rather than blindly suggesting training in whatever score is lowest, NEXUS computes:
$$\text{Bottleneck Priority}_k = w_{\text{gap}} \cdot \overline{\text{Gap}}_k + w_{\text{demand}} \cdot \text{MarketDemand}_k + w_{\text{impact}} \cdot \text{RoleUnlockImpact}_k$$

### C. 100% Local-First Data Privacy
In strict compliance with hackathon regulations:
- **Zero external API calls:** No calls to OpenAI, Claude, Gemini, or third-party cloud scrapers.
- **Zero data egress:** Resumes and candidate data never leave the local environment.
- Local NLP and parsing handled by `rapidfuzz`, `pypdf`, and `python-docx`.

---

## 5. Getting Started & Running Locally

### Prerequisites
- Python 3.11 or 3.12 (Tested on Python 3.12.10 Windows x64)

### Installation
Dependencies are already installed in this environment. To install in any new environment:
```bash
pip install -r requirements.txt
```

### Launch the Streamlit Application
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Run the Test Suite
```bash
python -m pytest tests/test_nexus.py -v
```
All 8 integration and unit tests execute in ~16 seconds with 100% pass rate.

### Run the CLI Analytical Pipeline
```bash
python backend/main.py
```

---

## 6. Page-by-Page Feature Tour

1. **Overview & Candidate Setup:** Executive briefing, live market KPIs, and 3 candidate intake modes (Demo Candidate, Interactive Sliders, Local Resume Upload).
2. **Market Intelligence:** Empirical salary-experience bubble charts, vacancy distributions, top hiring employers, and side-by-side dataset comparisons.
3. **Success Signals (JDS & SDS):** 5-fold cross-validated logistic regression models, decision trees, confusion matrices, and group comparison tables with significance tests.
4. **Capability & Opportunity Fit:** Radar capability mapping, granular per-role compatibility scoring, and bottleneck capability rankings.
5. **Counterfactual Transition Simulator:** Dynamic before-and-after simulation, Opportunity Expansion percentage, waterfall charts, quantitative explanation engine, and comparative intervention matrix.
6. **Data Governance & Audit:** Preprocessing logs, boundary enforcement audits, and local-first privacy compliance verification.
