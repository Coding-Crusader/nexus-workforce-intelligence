"""
NEXUS Opportunity Engine Module
Computes candidate-role compatibility scores, skill gap profiles,
and identifies high-leverage workforce transition bottlenecks.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from backend.data_loader import get_settings

class OpportunityEngine:
    """Evaluates multi-dimensional capability fit against empirical labor market standards."""

    def __init__(
        self,
        role_skill_matrix: Dict[str, Dict[str, float]],
        role_benchmarks_df: pd.DataFrame,
        skill_frequencies_df: pd.DataFrame,
        settings: Optional[Dict[str, Any]] = None
    ):
        self.role_profiles = role_skill_matrix
        self.benchmarks = role_benchmarks_df.set_index("job_title_clean") if "job_title_clean" in role_benchmarks_df.columns else role_benchmarks_df
        self.skill_freqs = skill_frequencies_df.set_index("skill_key") if "skill_key" in skill_frequencies_df.columns else skill_frequencies_df
        self.settings = settings or get_settings()

        engine_cfg = self.settings.get("opportunity_engine", {})
        self.compat_threshold = engine_cfg.get("compatibility_threshold", 0.65)
        self.stretch_threshold = engine_cfg.get("stretch_threshold", 0.50)
        self.weights = engine_cfg.get("scoring_weights", {
            "core_skill_match": 0.55,
            "secondary_skill_match": 0.25,
            "experience_alignment": 0.20
        })
        self.bottleneck_weights = engine_cfg.get("bottleneck_weighting", {
            "gap_weight": 0.35,
            "market_demand_weight": 0.35,
            "opportunity_impact_weight": 0.30
        })

    @staticmethod
    def calculate_evidence_confidence(
        base_score: float,
        evidence_strength: str,
        recency_months: float,
        half_life_months: float = 18.0
    ) -> Dict[str, float]:
        """Calculates evidence confidence adjusted for verification strength and skill half-life decay."""
        strength_multipliers = {
            "STRONG": 1.00,
            "MEDIUM": 0.85,
            "SUPPORTING": 0.75,
            "WEAK": 0.60
        }
        mult = strength_multipliers.get(str(evidence_strength).upper(), 0.70)
        decay = float(2.0 ** (-max(0.0, float(recency_months)) / half_life_months))
        decay = max(0.20, min(1.0, decay))
        final_confidence = round(float(base_score * mult * decay), 3)

        return {
            "base_score": round(base_score, 3),
            "strength_multiplier": mult,
            "recency_decay": round(decay, 3),
            "final_confidence": final_confidence
        }

    def evaluate_role_compatibility(
        self,
        capabilities: Dict[str, float],
        experience_years: float,
        role_name: str
    ) -> Dict[str, Any]:
        """Calculates granular compatibility and gap metrics for a specific role."""
        reqs = self.role_profiles.get(role_name, {})
        if not reqs:
            return {
                "role_name": role_name,
                "compatibility_score": 0.0,
                "status": "Blocked",
                "core_match": 0.0,
                "secondary_match": 0.0,
                "experience_match": 0.0,
                "skill_gaps": {}
            }

        core_skills = {k: v for k, v in reqs.items() if v >= 0.50}
        secondary_skills = {k: v for k, v in reqs.items() if 0.15 <= v < 0.50}

        # 1. Core skills coverage
        if core_skills:
            core_cov = sum(v * min(capabilities.get(k, 0.0) / v, 1.0) for k, v in core_skills.items()) / sum(core_skills.values())
        else:
            core_cov = 1.0

        # 2. Secondary skills coverage
        if secondary_skills:
            sec_cov = sum(v * min(capabilities.get(k, 0.0) / v, 1.0) for k, v in secondary_skills.items()) / sum(secondary_skills.values())
        else:
            sec_cov = core_cov

        # 3. Experience requirement alignment
        req_exp = float(self.benchmarks.loc[role_name, "min_experience_years"]) if role_name in self.benchmarks.index else 2.0
        exp_cov = min(1.0, experience_years / max(req_exp, 0.5))

        w_core = self.weights.get("core_skill_match", 0.55)
        w_sec = self.weights.get("secondary_skill_match", 0.25)
        w_exp = self.weights.get("experience_alignment", 0.20)

        raw_score = (w_core * core_cov) + (w_sec * sec_cov) + (w_exp * exp_cov)

        # Deficit penalty if candidate has critical gap in dominant core competency (req >= 0.70 & cap < 0.40)
        for k, v in core_skills.items():
            if v >= 0.70 and capabilities.get(k, 0.0) < 0.40:
                penalty = (0.40 - capabilities.get(k, 0.0)) * 0.45
                raw_score = max(0.0, raw_score - penalty)

        score = round(float(raw_score), 3)

        if score >= self.compat_threshold:
            status = "Reachable"
        elif score >= self.stretch_threshold:
            status = "Stretch"
        else:
            status = "Blocked"

        # Calculate exact skill gaps
        gaps = {}
        for k, req_val in reqs.items():
            cand_val = capabilities.get(k, 0.0)
            if req_val > cand_val:
                gaps[k] = round(req_val - cand_val, 3)

        # Benchmark context
        avg_sal = float(self.benchmarks.loc[role_name, "avg_salary_lakhs"]) if role_name in self.benchmarks.index else 0.0
        vacancies = int(self.benchmarks.loc[role_name, "total_vacancies"]) if role_name in self.benchmarks.index else 0

        return {
            "role_name": role_name,
            "compatibility_score": score,
            "compatibility_pct": round(score * 100, 1),
            "status": status,
            "core_match_pct": round(core_cov * 100, 1),
            "secondary_match_pct": round(sec_cov * 100, 1),
            "experience_match_pct": round(exp_cov * 100, 1),
            "required_experience": req_exp,
            "average_salary_lakhs": avg_sal,
            "market_vacancies": vacancies,
            "skill_gaps": gaps
        }

    def evaluate_candidate_landscape(
        self,
        capabilities: Dict[str, float],
        experience_years: float
    ) -> Dict[str, Any]:
        """Maps entire opportunity landscape for a given candidate profile."""
        results = []
        for role_name in self.role_profiles.keys():
            res = self.evaluate_role_compatibility(capabilities, experience_years, role_name)
            results.append(res)

        df_roles = pd.DataFrame(results).sort_values(by="compatibility_score", ascending=False).reset_index(drop=True)

        reachable = df_roles[df_roles["status"] == "Reachable"]
        stretch = df_roles[df_roles["status"] == "Stretch"]
        blocked = df_roles[df_roles["status"] == "Blocked"]

        reachable_vacancies = int(reachable["market_vacancies"].sum())
        avg_reachable_salary = round(float(reachable["average_salary_lakhs"].mean()), 2) if len(reachable) > 0 else 0.0

        return {
            "roles_table": df_roles,
            "reachable_count": len(reachable),
            "stretch_count": len(stretch),
            "blocked_count": len(blocked),
            "total_roles_evaluated": len(df_roles),
            "reachable_vacancies": reachable_vacancies,
            "average_reachable_salary": avg_reachable_salary,
            "reachable_roles": reachable["role_name"].tolist(),
            "stretch_roles": stretch["role_name"].tolist(),
            "blocked_roles": blocked["role_name"].tolist()
        }

    def detect_bottlenecks(
        self,
        capabilities: Dict[str, float],
        experience_years: float,
        simulated_delta: float = 0.30
    ) -> pd.DataFrame:
        """
        Pinpoints high-leverage capability bottlenecks across blocked and stretch roles.
        Considers candidate gap, market demand frequency, and opportunity expansion impact.
        """
        current_landscape = self.evaluate_candidate_landscape(capabilities, experience_years)
        current_reachable = set(current_landscape["reachable_roles"])
        non_reachable = set(current_landscape["stretch_roles"] + current_landscape["blocked_roles"])

        bottleneck_data = []
        all_skills = list(capabilities.keys())

        for skill in all_skills:
            cand_lvl = capabilities.get(skill, 0.0)

            # 1. Average gap across non-reachable roles that actually need this skill
            gaps = []
            for role in non_reachable:
                req_val = self.role_profiles.get(role, {}).get(skill, 0.0)
                if req_val > cand_lvl:
                    gaps.append(req_val - cand_lvl)
            avg_gap = float(np.mean(gaps)) if gaps else 0.0

            # 2. Market demand frequency weight
            demand_count = float(self.skill_freqs.loc[skill, "demand_count"]) if skill in self.skill_freqs.index else 0.0
            max_demand = float(self.skill_freqs["demand_count"].max()) if not self.skill_freqs.empty else 1.0
            demand_weight = demand_count / max(max_demand, 1.0)

            # 3. Counterfactual impact of improving this single skill
            hypo_caps = capabilities.copy()
            hypo_caps[skill] = min(1.0, cand_lvl + simulated_delta)
            hypo_landscape = self.evaluate_candidate_landscape(hypo_caps, experience_years)
            hypo_reachable = set(hypo_landscape["reachable_roles"])
            newly_unlocked = hypo_reachable - current_reachable
            unlocked_count = len(newly_unlocked)
            vacancy_expansion = hypo_landscape["reachable_vacancies"] - current_landscape["reachable_vacancies"]

            # Normalized impact score (up to 4 unlocked roles = 1.0)
            impact_score = min(1.0, unlocked_count / 3.0)

            # Composite Priority Score
            w_gap = self.bottleneck_weights.get("gap_weight", 0.35)
            w_dem = self.bottleneck_weights.get("market_demand_weight", 0.35)
            w_imp = self.bottleneck_weights.get("opportunity_impact_weight", 0.30)

            priority_score = (w_gap * avg_gap) + (w_dem * demand_weight) + (w_imp * impact_score)

            display_name = skill.replace("_", " ").title()
            bottleneck_data.append({
                "skill_key": skill,
                "display_name": display_name,
                "current_level": round(cand_lvl, 2),
                "average_gap_in_target_roles": round(avg_gap, 2),
                "market_demand_weight": round(demand_weight, 2),
                "unlocked_roles_count": unlocked_count,
                "newly_unlocked_roles": list(newly_unlocked),
                "vacancy_expansion": vacancy_expansion,
                "priority_score": round(priority_score, 3)
            })

        df_bn = pd.DataFrame(bottleneck_data).sort_values(by="priority_score", ascending=False).reset_index(drop=True)
        return df_bn
