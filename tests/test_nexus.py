"""
Comprehensive Test Suite for NEXUS Workforce Transition Intelligence
Tests data loader, cleaner, skill analyzer, market analyzer, ML models, and simulation engine.
"""

import pytest
import pandas as pd
import numpy as np

from backend.data_loader import (
    load_all_raw_datasets, get_settings, get_skill_aliases,
    get_interventions, get_demo_candidate
)
from backend.data_cleaner import DataCleaner, parse_experience_string, parse_salary_lakhs
from backend.skill_analysis import SkillAnalyzer
from backend.market_analysis import MarketAnalyzer
from backend.success_analysis import SuccessAnalyzer
from backend.opportunity_engine import OpportunityEngine
from backend.simulation import TransitionSimulator

# 1. Config & Data Loader Tests
def test_config_loading():
    settings = get_settings()
    assert "opportunity_engine" in settings
    assert "simulation" in settings

    aliases = get_skill_aliases()
    assert "python" in aliases
    assert "machine_learning" in aliases

    interventions = get_interventions()
    assert len(interventions) >= 3

    candidate = get_demo_candidate()
    assert "capabilities" in candidate
    assert candidate["years_of_experience"] > 0

def test_data_loader_raw():
    raw_data = load_all_raw_datasets()
    assert "analytics_jobs" in raw_data
    assert "datascience_jobs" in raw_data
    assert "jds_skills" in raw_data
    assert "sds_personality" in raw_data

    assert len(raw_data["analytics_jobs"]) > 15000
    assert len(raw_data["datascience_jobs"]) > 1500
    assert len(raw_data["jds_skills"]) > 100
    assert len(raw_data["sds_personality"]) > 100

# 2. Data Cleaner Tests
def test_cleaner_parsers():
    mn, mx, avg = parse_experience_string("5-10 yrs")
    assert mn == 5.0 and mx == 10.0 and avg == 7.5

    sal = parse_salary_lakhs("12.8L")
    assert sal == 12.8

def test_cleaner_pipeline():
    raw_data = load_all_raw_datasets()
    cleaner = DataCleaner()
    cleaned_data, reports = cleaner.clean_all(raw_data)

    for k in ["analytics_jobs", "datascience_jobs", "jds_skills", "sds_personality"]:
        assert k in cleaned_data
        assert k in reports
        assert reports[k]["cleaned_rows"] == reports[k]["original_rows"]

# 3. Skill Analysis Tests
def test_skill_analyzer():
    analyzer = SkillAnalyzer()
    norm = analyzer.normalize_single_skill("ml")
    assert norm == "machine_learning"
    norm_py = analyzer.normalize_single_skill("python scripting")
    assert norm_py == "python"

    raw_data = load_all_raw_datasets()
    cleaner = DataCleaner()
    cleaned_aj, _ = cleaner.clean_analytics_jobs(raw_data["analytics_jobs"])

    skill_freqs = analyzer.compute_skill_frequencies(cleaned_aj)
    assert not skill_freqs.empty
    assert "demand_count" in skill_freqs.columns

    role_matrix = analyzer.compute_role_skill_matrix(cleaned_aj)
    assert len(role_matrix) >= 5

# 4. Market Analysis Tests
def test_market_analyzer():
    raw_data = load_all_raw_datasets()
    cleaner = DataCleaner()
    cleaned_aj, _ = cleaner.clean_analytics_jobs(raw_data["analytics_jobs"])
    cleaned_ds, _ = cleaner.clean_datascience_jobs(raw_data["datascience_jobs"])

    market = MarketAnalyzer(cleaned_aj, cleaned_ds)
    headlines = market.get_headline_metrics()
    assert headlines["total_market_vacancies"] > 50000
    assert headlines["average_market_salary_lakhs"] > 5.0

    benchmarks = market.get_role_demand_and_salary_benchmarks()
    assert len(benchmarks) >= 8

# 5. Success Analysis Tests
def test_success_analysis_models():
    raw_data = load_all_raw_datasets()
    cleaner = DataCleaner()
    cleaned_jds, _ = cleaner.clean_jds_skills(raw_data["jds_skills"])
    cleaned_sds, _ = cleaner.clean_sds_personality(raw_data["sds_personality"])

    analyzer = SuccessAnalyzer(cleaned_jds, cleaned_sds)
    jds_res = analyzer.analyze_jds()
    assert jds_res["logistic_regression"]["accuracy"] > 0.70
    assert jds_res["logistic_regression"]["roc_auc"] > 0.70

    sds_res = analyzer.analyze_sds()
    assert sds_res["logistic_regression"]["accuracy"] > 0.80
    assert sds_res["logistic_regression"]["roc_auc"] > 0.80

# 6. Opportunity Engine & Simulation Tests
def test_opportunity_and_simulation():
    raw_data = load_all_raw_datasets()
    cleaner = DataCleaner()
    cleaned_data, _ = cleaner.clean_all(raw_data)

    skill_analyzer = SkillAnalyzer()
    skill_freqs = skill_analyzer.compute_skill_frequencies(cleaned_data["analytics_jobs"])
    role_profiles = skill_analyzer.compute_role_skill_matrix(cleaned_data["analytics_jobs"])

    market = MarketAnalyzer(cleaned_data["analytics_jobs"], cleaned_data["datascience_jobs"])
    benchmarks = market.get_role_demand_and_salary_benchmarks()

    engine = OpportunityEngine(role_profiles, benchmarks, skill_freqs)
    candidate = get_demo_candidate()

    landscape = engine.evaluate_candidate_landscape(candidate["capabilities"], candidate["years_of_experience"])
    assert landscape["reachable_count"] > 0
    assert landscape["total_roles_evaluated"] >= 8

    # Test Bottleneck detection
    df_bn = engine.detect_bottlenecks(candidate["capabilities"], candidate["years_of_experience"])
    assert not df_bn.empty
    assert "priority_score" in df_bn.columns

    # Test Simulation
    sim = TransitionSimulator(engine)
    interventions = get_interventions()
    first_id = interventions[0]["id"]
    sim_res = sim.simulate_intervention(
        candidate["capabilities"],
        candidate["years_of_experience"],
        first_id
    )
    assert sim_res["future_reachable_count"] >= sim_res["baseline_reachable_count"]
    assert "opportunity_expansion_pct" in sim_res
    assert len(sim_res["explanation"]["bullet_reasons"]) > 0

    # Test Evidence Confidence & Skill Half-Life calculation (Slide 04)
    conf = OpportunityEngine.calculate_evidence_confidence(
        base_score=0.85,
        evidence_strength="STRONG",
        recency_months=2,
        half_life_months=18
    )
    assert 0.0 < conf["final_confidence"] <= 1.0


def test_candidate_database_and_evidence_pipeline(tmp_path):
    """Tests candidate relational database entity and multi-source evidence scoring."""
    from backend.database import CandidateDB
    from backend.evidence_ingestion import GitHubConnector, SkillPlatformConnector
    from backend.evidence_scoring import EvidenceScorer
    import pandas as pd
    import json

    db_file = tmp_path / "test_nexus.db"
    db = CandidateDB(db_file)
    cand = db.get_or_create_candidate("student@university.edu", name="Rohan Das", title="Student", years_exp=1.0)
    assert cand["email"] == "student@university.edu"

    # Skill platform ingestion
    sp_conn = SkillPlatformConnector()
    df_sample = pd.DataFrame([
        {"problem_name": "Two Sum", "language": "Python", "topic": "Arrays", "difficulty": "Easy", "result": "Accepted", "date": "2026-09-01"},
        {"problem_name": "Merge Intervals", "language": "Python", "topic": "Arrays", "difficulty": "Medium", "result": "Accepted", "date": "2026-09-10"},
        {"problem_name": "LRU Cache", "language": "C++", "topic": "Data Structures", "difficulty": "Hard", "result": "Accepted", "date": "2026-09-15"},
    ])
    sp_evidence = sp_conn.process_csv_or_df(df_sample, "LeetCode")
    src_id = db.register_source(cand["candidate_id"], sp_evidence["source_type"], sp_evidence["source_identifier"])
    db.add_evidence(
        cand["candidate_id"], src_id, sp_evidence["evidence_type"],
        sp_evidence["title"], sp_evidence["description"], sp_evidence["verification_strength"],
        sp_evidence["base_score"], sp_evidence["recency_months"], sp_evidence["skill_events"]
    )

    events = db.get_skill_events_for_candidate(cand["candidate_id"])
    assert len(events) >= 2

    scorer = EvidenceScorer()
    scored = scorer.score_candidate_events(events)
    assert "python" in scored["capabilities"]
    assert len(scored["provenance_explanations"]["python"]) > 0

    db.save_derived_skills(cand["candidate_id"], scored["capabilities"], scored["confidence_scores"], scored["provenance_explanations"])
    loaded = db.get_derived_skills(cand["candidate_id"])
    assert "python" in loaded
    assert loaded["python"]["score"] > 0.30

