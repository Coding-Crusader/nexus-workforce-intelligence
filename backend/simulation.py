"""
NEXUS Counterfactual Transition Simulator Module
Deterministic simulation of targeted upskilling interventions,
opportunity landscape recalculation, expansion metrics, and quantitative explanations.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from backend.data_loader import get_interventions, get_settings
from backend.opportunity_engine import OpportunityEngine

class TransitionSimulator:
    """Simulates counterfactual workforce interventions and computes opportunity expansion."""

    def __init__(self, engine: OpportunityEngine, interventions: Optional[List[Dict[str, Any]]] = None):
        self.engine = engine
        self.interventions = interventions or get_interventions()
        self.intervention_map = {item["id"]: item for item in self.interventions}

    def simulate_intervention(
        self,
        current_capabilities: Dict[str, float],
        experience_years: float,
        intervention_id: str,
        custom_boost_factor: float = 1.0
    ) -> Dict[str, Any]:
        """
        Executes deterministic counterfactual simulation:
        1. Evaluates baseline opportunity landscape.
        2. Applies intervention capability deltas.
        3. Evaluates counterfactual future landscape.
        4. Calculates Opportunity Expansion rate, delta scores, and newly unlocked roles.
        5. Formulates quantitative, evidence-backed explanations.
        """
        if intervention_id not in self.intervention_map:
            raise ValueError(f"Unknown intervention ID: {intervention_id}")

        intervention = self.intervention_map[intervention_id]
        skills_affected = intervention.get("skills_affected", {})

        # 1. Baseline state
        baseline_landscape = self.engine.evaluate_candidate_landscape(current_capabilities, experience_years)
        baseline_roles_df = baseline_landscape["roles_table"].set_index("role_name")
        baseline_reachable = set(baseline_landscape["reachable_roles"])
        baseline_vacancies = baseline_landscape["reachable_vacancies"]
        baseline_avg_salary = baseline_landscape["average_reachable_salary"]

        # 2. Counterfactual Future State
        future_capabilities = current_capabilities.copy()
        applied_deltas = {}
        for skill_key, raw_boost in skills_affected.items():
            boost = round(raw_boost * custom_boost_factor, 3)
            current_val = future_capabilities.get(skill_key, 0.0)
            new_val = min(1.0, round(current_val + boost, 3))
            future_capabilities[skill_key] = new_val
            applied_deltas[skill_key] = {
                "before": current_val,
                "after": new_val,
                "boost": round(new_val - current_val, 3)
            }

        # Applied project experience from intervention capstone:
        # Completing an intensive 10-12 week curriculum with an end-to-end capstone project
        # provides verified project implementation, elevating freshers/students towards entry-level readiness
        effective_future_exp = max(experience_years, 0.75) if experience_years < 1.0 else experience_years

        # 3. Future landscape
        future_landscape = self.engine.evaluate_candidate_landscape(future_capabilities, effective_future_exp)
        future_roles_df = future_landscape["roles_table"].set_index("role_name")
        future_reachable = set(future_landscape["reachable_roles"])
        future_stretch = set(future_landscape.get("stretch_roles", []))
        future_vacancies = future_landscape["reachable_vacancies"]
        future_avg_salary = future_landscape["average_reachable_salary"]

        # 4. Comparative Metrics
        newly_reachable = sorted(list(future_reachable - baseline_reachable))
        baseline_stretch = set(baseline_landscape.get("stretch_roles", []))
        newly_stretch = sorted(list((future_stretch - baseline_stretch) - baseline_reachable))
        retained_reachable_roles = sorted(list(future_reachable.intersection(baseline_reachable)))
        remaining_blocked_roles = sorted(list(set(future_landscape["blocked_roles"])))
        remaining_stretch_roles = sorted(list(set(future_landscape["stretch_roles"])))

        n_before = len(baseline_reachable)
        n_after = len(future_reachable)

        # Opportunity Expansion & Unlocked Roles Calculation
        if n_before > 0:
            opportunity_expansion_ratio = (n_after - n_before) / n_before
            opportunity_expansion_pct = round(opportunity_expansion_ratio * 100.0, 1)
            vacancy_growth = future_vacancies - baseline_vacancies
            salary_growth_lakhs = round(future_avg_salary - baseline_avg_salary, 2)
            newly_unlocked_roles = newly_reachable
        else:
            # Cold-start / Fresher candidate with 0 baseline reachable roles
            if n_after > 0:
                opportunity_expansion_pct = round(n_after * 100.0, 1)
                opportunity_expansion_ratio = float(n_after)
                vacancy_growth = future_vacancies
                salary_growth_lakhs = round(future_avg_salary, 2)
                newly_unlocked_roles = newly_reachable
            elif newly_stretch:
                # Interventions unlocking near-reachable / stretch entry career tracks
                unlocked_vacancies = sum(int(future_roles_df.loc[r, "market_vacancies"]) for r in newly_stretch)
                unlocked_sal = float(np.mean([future_roles_df.loc[r, "average_salary_lakhs"] for r in newly_stretch]))
                opportunity_expansion_pct = round(len(newly_stretch) * 100.0, 1)
                opportunity_expansion_ratio = float(len(newly_stretch))
                vacancy_growth = unlocked_vacancies
                salary_growth_lakhs = round(unlocked_sal, 2)
                newly_unlocked_roles = newly_stretch
            else:
                opportunity_expansion_ratio = 0.0
                opportunity_expansion_pct = 0.0
                vacancy_growth = 0
                salary_growth_lakhs = 0.0
                newly_unlocked_roles = []

        # Per-role transition comparison table
        comparison_records = []
        for role_name in baseline_roles_df.index:
            b_row = baseline_roles_df.loc[role_name]
            f_row = future_roles_df.loc[role_name]

            score_before = float(b_row["compatibility_score"])
            score_after = float(f_row["compatibility_score"])
            delta_score = round(score_after - score_before, 3)
            status_before = str(b_row["status"])
            status_after = str(f_row["status"])

            is_new = role_name in newly_unlocked_roles

            comparison_records.append({
                "role_name": role_name,
                "score_before": round(score_before * 100, 1),
                "score_after": round(score_after * 100, 1),
                "delta_score_pct": round(delta_score * 100, 1),
                "status_before": status_before,
                "status_after": status_after,
                "is_newly_unlocked": is_new,
                "average_salary_lakhs": float(b_row["average_salary_lakhs"]),
                "market_vacancies": int(b_row["market_vacancies"])
            })

        df_comparison = pd.DataFrame(comparison_records).sort_values(by="delta_score_pct", ascending=False).reset_index(drop=True)

        # 5. Remaining Bottlenecks in Future State
        future_bottlenecks = self.engine.detect_bottlenecks(future_capabilities, experience_years)

        # 6. Formulate Data-Backed Explanation
        explanation = self._generate_explanation(
            intervention=intervention,
            applied_deltas=applied_deltas,
            newly_unlocked_roles=newly_unlocked_roles,
            df_comparison=df_comparison,
            opportunity_expansion_pct=opportunity_expansion_pct,
            vacancy_growth=vacancy_growth,
            salary_growth_lakhs=salary_growth_lakhs
        )

        return {
            "intervention": intervention,
            "applied_deltas": applied_deltas,
            "baseline_capabilities": current_capabilities,
            "future_capabilities": future_capabilities,
            "baseline_reachable_count": n_before,
            "future_reachable_count": n_after,
            "opportunity_expansion_pct": opportunity_expansion_pct,
            "opportunity_expansion_ratio": round(opportunity_expansion_ratio, 3),
            "newly_unlocked_roles": newly_unlocked_roles,
            "retained_reachable_roles": retained_reachable_roles,
            "remaining_stretch_roles": remaining_stretch_roles,
            "remaining_blocked_roles": remaining_blocked_roles,
            "vacancy_growth": vacancy_growth,
            "baseline_vacancies": baseline_vacancies,
            "future_vacancies": future_vacancies,
            "salary_growth_lakhs": salary_growth_lakhs,
            "baseline_avg_salary": baseline_avg_salary,
            "future_avg_salary": future_avg_salary,
            "roles_comparison_table": df_comparison,
            "future_bottlenecks": future_bottlenecks,
            "explanation": explanation
        }

    def simulate_all_interventions(
        self,
        current_capabilities: Dict[str, float],
        experience_years: float
    ) -> List[Dict[str, Any]]:
        """Simulates all available interventions for comparative ranking and selection."""
        results = []
        for intv in self.interventions:
            sim_res = self.simulate_intervention(current_capabilities, experience_years, intv["id"])
            results.append(sim_res)

        # Sort by opportunity expansion and vacancy growth
        results.sort(key=lambda x: (x["opportunity_expansion_pct"], x["vacancy_growth"]), reverse=True)
        return results

    def _generate_explanation(
        self,
        intervention: Dict[str, Any],
        applied_deltas: Dict[str, Dict[str, float]],
        newly_unlocked_roles: List[str],
        df_comparison: pd.DataFrame,
        opportunity_expansion_pct: float,
        vacancy_growth: int,
        salary_growth_lakhs: float
    ) -> Dict[str, Any]:
        """Synthesizes structured, data-grounded causal explanation for the simulation."""
        skills_summary = ", ".join([f"{k.replace('_', ' ').title()} (+{v['boost']:.2f})" for k, v in applied_deltas.items()])
        
        top_improving_roles = df_comparison.head(3)[["role_name", "delta_score_pct"]].to_dict(orient="records")

        reasons = [
            f"Targeted Skill Resolution: Addresses candidate gaps in {skills_summary}.",
            f"Opportunity Expansion: Achieves a +{opportunity_expansion_pct:.1f}% expansion in qualified roles across the empirical labor market.",
            f"Market Reach: Expands accessible job vacancies by +{vacancy_growth:,} positions across analyzed hiring enterprises.",
            f"Compensation Trajectory: Lifts the average reachable benchmark salary by +{salary_growth_lakhs:+.2f}L."
        ]

        if newly_unlocked_roles:
            unlocked_str = ", ".join(newly_unlocked_roles)
            reasons.append(f"Newly Reachable Career Tracks: Successfully transitions candidate into: {unlocked_str}.")
        else:
            reasons.append("Consolidation Impact: Deepens competitive margin in existing roles, raising readiness for promotion.")

        return {
            "headline": f"Why this intervention matters: {intervention['name']}",
            "rationale_statement": intervention.get("rationale", ""),
            "bullet_reasons": reasons,
            "top_improving_roles": top_improving_roles,
            "newly_unlocked_roles": newly_unlocked_roles
        }
