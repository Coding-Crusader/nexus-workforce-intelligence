import sys
sys.path.insert(0, r'C:\Coding_Crusader\NEXUS')

import json
from backend.data_loader import (
    load_analytics_jobs, load_datascience_jobs, get_demo_candidate,
    get_interventions, get_settings
)
from backend.data_cleaner import DataCleaner
from backend.skill_analysis import SkillAnalyzer
from backend.market_analysis import MarketAnalyzer
from backend.opportunity_engine import OpportunityEngine

cleaner = DataCleaner()
df_aj, _ = cleaner.clean_analytics_jobs(load_analytics_jobs())
df_ds, _ = cleaner.clean_datascience_jobs(load_datascience_jobs())

analyzer = SkillAnalyzer()
role_profiles = analyzer.compute_role_skill_matrix(df_aj)
skill_freqs = analyzer.compute_skill_frequencies(df_aj)

market_analyzer = MarketAnalyzer(df_aj, df_ds)
benchmarks = market_analyzer.get_role_demand_and_salary_benchmarks()

engine = OpportunityEngine(role_profiles, benchmarks, skill_freqs)
candidate = get_demo_candidate()
interventions = get_interventions()

current_caps = candidate['capabilities']
exp = candidate['years_of_experience']

current_landscape = engine.evaluate_candidate_landscape(current_caps, exp)
print('Current Reachable Roles:', current_landscape['reachable_roles'])

for intv in interventions:
    print('='*50)
    print('Testing Intervention:', intv['name'])
    future_caps = current_caps.copy()
    for s, boost in intv['skills_affected'].items():
        future_caps[s] = min(1.0, future_caps.get(s, 0.0) + boost)
    
    future_landscape = engine.evaluate_candidate_landscape(future_caps, exp)
    
    before_reach = set(current_landscape['reachable_roles'])
    after_reach = set(future_landscape['reachable_roles'])
    newly_unlocked = after_reach - before_reach
    
    n_before = len(before_reach)
    n_after = len(after_reach)
    exp_rate = ((n_after - n_before) / max(n_before, 1)) * 100.0 if n_before > 0 else (100.0 if n_after > 0 else 0.0)
    
    print(f'Reachable: {n_before} -> {n_after} (Expansion: +{exp_rate:.1f}%)')
    print('Newly Unlocked:', list(newly_unlocked))
    print('Reachable Vacancies: ', current_landscape['reachable_vacancies'], '->', future_landscape['reachable_vacancies'])
    print('Avg Salary Potential: ', current_landscape['average_reachable_salary'], 'L ->', future_landscape['average_reachable_salary'], 'L')
