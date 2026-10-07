"""
NEXUS Backend Pipeline Orchestrator & CLI Runner
Runs end-to-end data ingestion, cleaning, empirical modeling, and simulation.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

def run_nexus_pipeline():
    """Executes full analytical and transition intelligence workflow."""
    print("=" * 70)
    print(" NEXUS: Workforce Transition Intelligence — Pipeline Execution")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/6] Loading raw hackathon competition datasets...")
    raw_data = load_all_raw_datasets()
    print("  -> Loaded Analytics Jobs, Data Science Jobs, JDS Skills, SDS Personality.")

    # 2. Clean and Audit Data
    print("\n[2/6] Running DataCleaner preprocessing and quality audit...")
    cleaner = DataCleaner()
    cleaned_data, quality_reports = cleaner.clean_all(raw_data)
    for name, r in quality_reports.items():
        print(f"  -> {r['dataset_name']}: {r['cleaned_rows']} rows verified ({len(r['operations'])} operations).")

    # 3. Market Intelligence
    print("\n[3/6] Computing labor market intelligence and benchmarks...")
    market_analyzer = MarketAnalyzer(cleaned_data["analytics_jobs"], cleaned_data["datascience_jobs"])
    headline = market_analyzer.get_headline_metrics()
    benchmarks = market_analyzer.get_role_demand_and_salary_benchmarks()
    print(f"  -> Analyzed {headline['total_market_vacancies']:,} job vacancies across {headline['unique_companies_represented']} companies.")
    print(f"  -> Overall Average Market Salary: {headline['average_market_salary_lakhs']}L INR.")

    # 4. Skill Taxonomy & Role Profiling
    print("\n[4/6] Normalizing skills and generating empirical role profiles...")
    skill_analyzer = SkillAnalyzer()
    skill_freqs = skill_analyzer.compute_skill_frequencies(cleaned_data["analytics_jobs"])
    role_profiles = skill_analyzer.compute_role_skill_matrix(cleaned_data["analytics_jobs"])
    print(f"  -> Evaluated {len(skill_freqs)} canonical skill families across {len(role_profiles)} standardized roles.")

    # 5. Success Signals Analysis (JDS & SDS ML)
    print("\n[5/6] Training empirical classifiers for JDS skills and SDS traits...")
    success_analyzer = SuccessAnalyzer(cleaned_data["jds_skills"], cleaned_data["sds_personality"])
    jds_results = success_analyzer.analyze_jds()
    sds_results = success_analyzer.analyze_sds()
    print(f"  -> JDS Salary Hike Classifier: 5-Fold CV Acc = {jds_results['logistic_regression']['cv_accuracy_mean']*100:.1f}%, ROC-AUC = {jds_results['logistic_regression']['roc_auc']:.3f}")
    print(f"  -> SDS Success Classifier:     5-Fold CV Acc = {sds_results['logistic_regression']['cv_accuracy_mean']*100:.1f}%, ROC-AUC = {sds_results['logistic_regression']['roc_auc']:.3f}")

    # 6. Opportunity Engine & Counterfactual Simulation
    print("\n[6/6] Initializing Opportunity Engine & Simulating Demo Candidate...")
    engine = OpportunityEngine(role_profiles, benchmarks, skill_freqs)
    candidate = get_demo_candidate()
    print(f"  -> Loaded Demo Candidate: {candidate['name']} ({candidate['current_title']})")

    baseline_landscape = engine.evaluate_candidate_landscape(candidate["capabilities"], candidate["years_of_experience"])
    print(f"  -> Baseline Reachable Roles: {baseline_landscape['reachable_count']} of {baseline_landscape['total_roles_evaluated']}")
    print(f"  -> Baseline Reachable Roles: {baseline_landscape['reachable_roles']}")

    simulator = TransitionSimulator(engine)
    sim_results = simulator.simulate_all_interventions(candidate["capabilities"], candidate["years_of_experience"])

    print("\n=== TOP COUNTERFACTUAL INTERVENTIONS ===")
    for rank, res in enumerate(sim_results, start=1):
        intv = res["intervention"]
        print(f"\n#{rank}: {intv['name']}")
        print(f"    Opportunity Expansion: +{res['opportunity_expansion_pct']:.1f}% ({res['baseline_reachable_count']} -> {res['future_reachable_count']} roles)")
        print(f"    Newly Unlocked Roles:  {res['newly_unlocked_roles']}")
        print(f"    Vacancy Expansion:     +{res['vacancy_growth']:,} accessible jobs")
        print(f"    Avg Salary Potential:  {res['baseline_avg_salary']}L -> {res['future_avg_salary']}L (+{res['salary_growth_lakhs']:+.2f}L)")

    print("\n" + "=" * 70)
    print(" Pipeline completed successfully with 100% deterministic reproducibility.")
    print("=" * 70)
    return {
        "cleaned_data": cleaned_data,
        "quality_reports": quality_reports,
        "market_metrics": headline,
        "benchmarks": benchmarks,
        "skill_freqs": skill_freqs,
        "role_profiles": role_profiles,
        "jds_results": jds_results,
        "sds_results": sds_results,
        "candidate": candidate,
        "simulation_results": sim_results
    }

if __name__ == "__main__":
    run_nexus_pipeline()
