"""
NEXUS Skill Analysis Module
High-performance skill extraction, normalization via alias maps,
role-skill profiling, and skill co-occurrence matrices.
"""

from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

from backend.data_loader import get_skill_aliases

class SkillAnalyzer:
    """Performs empirical skill normalization, frequency tracking, and role profiling with fast vectorized/dict execution."""

    def __init__(self, aliases: Optional[Dict[str, List[str]]] = None):
        self.aliases = aliases or get_skill_aliases()
        self._build_lookup_maps()

    def _build_lookup_maps(self):
        """Constructs fast direct and inverted alias maps."""
        self.alias_to_canonical = {}
        for canonical, variants in self.aliases.items():
            self.alias_to_canonical[canonical.lower()] = canonical
            for v in variants:
                self.alias_to_canonical[v.lower().strip()] = canonical

    def normalize_single_skill(self, raw_token: str) -> Optional[str]:
        """Fast O(1) canonical skill lookup."""
        if not raw_token or pd.isna(raw_token):
            return None
        token = str(raw_token).lower().strip().rstrip(".")
        return self.alias_to_canonical.get(token, None)

    def extract_canonical_skills_for_row(self, skills_list: List[str]) -> List[str]:
        """Extracts unique canonical skills from a job posting's skill list."""
        canonical_set = set()
        for item in skills_list:
            norm = self.normalize_single_skill(item)
            if norm:
                canonical_set.add(norm)
        return sorted(list(canonical_set))

    def compute_skill_frequencies(self, df_analytics: pd.DataFrame) -> pd.DataFrame:
        """
        Computes frequency, job coverage %, and role presence for all canonical skills.
        Runs in milliseconds by checking pre-extracted skills or vectorized processing.
        """
        total_jobs = len(df_analytics)
        skill_counts = Counter()
        skill_to_roles = {k: Counter() for k in self.aliases.keys()}

        # Check if canonical_skills_list is already pre-extracted in dataframe
        if "canonical_skills_list" in df_analytics.columns:
            for _, row in df_analytics.iterrows():
                canonical_skills = row["canonical_skills_list"]
                role = row.get("role_family", "Other")
                for s in canonical_skills:
                    skill_counts[s] += 1
                    skill_to_roles[s][role] += 1
        else:
            for _, row in df_analytics.iterrows():
                skills = row.get("skills_list", [])
                canonical_skills = self.extract_canonical_skills_for_row(skills)
                role = row.get("role_family", "Other")
                for s in canonical_skills:
                    skill_counts[s] += 1
                    skill_to_roles[s][role] += 1

        records = []
        for canonical_key in self.aliases.keys():
            count = skill_counts[canonical_key]
            pct = (count / total_jobs * 100.0) if total_jobs > 0 else 0.0
            top_role = skill_to_roles[canonical_key].most_common(1)
            primary_role = top_role[0][0] if top_role else "Various"

            display_name = canonical_key.replace("_", " ").title()
            records.append({
                "skill_key": canonical_key,
                "display_name": display_name,
                "demand_count": count,
                "job_coverage_pct": round(pct, 2),
                "primary_associated_role": primary_role
            })

        df_res = pd.DataFrame(records).sort_values(by="demand_count", ascending=False).reset_index(drop=True)
        return df_res

    def compute_role_skill_matrix(self, df_analytics: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Computes the relative skill intensity profiles (normalized 0.0 to 1.0)
        for each role family based on actual job vacancy requirements.
        """
        role_profiles = {}
        has_precalc = "canonical_skills_list" in df_analytics.columns

        for role, group in df_analytics.groupby("role_family"):
            if role in ["Other / Unspecified", "Other / Domain Specialist"]:
                continue

            n_jobs = len(group)
            if n_jobs < 10:
                continue

            counts = Counter()
            if has_precalc:
                for canonical_skills in group["canonical_skills_list"]:
                    for s in canonical_skills:
                        counts[s] += 1
            else:
                for skills in group["skills_list"]:
                    canonical_skills = self.extract_canonical_skills_for_row(skills)
                    for s in canonical_skills:
                        counts[s] += 1

            weights = {}
            for canonical_key in self.aliases.keys():
                raw_prob = counts[canonical_key] / n_jobs
                weights[canonical_key] = round(raw_prob, 3)

            max_p = max(weights.values()) if weights.values() else 1.0
            scaled_profile = {k: round(v / max_p, 3) if max_p > 0 else 0.0 for k, v in weights.items()}
            role_profiles[role] = scaled_profile

        return role_profiles

    def compute_skill_cooccurrence(self, df_analytics: pd.DataFrame) -> pd.DataFrame:
        """Calculates co-occurrence matrix between top canonical skills."""
        canonical_keys = sorted(list(self.aliases.keys()))
        matrix = pd.DataFrame(0, index=canonical_keys, columns=canonical_keys)
        has_precalc = "canonical_skills_list" in df_analytics.columns

        for _, row in df_analytics.iterrows():
            canonical_skills = row["canonical_skills_list"] if has_precalc else self.extract_canonical_skills_for_row(row.get("skills_list", []))
            for i, s1 in enumerate(canonical_skills):
                for s2 in canonical_skills[i:]:
                    matrix.loc[s1, s2] += 1
                    if s1 != s2:
                        matrix.loc[s2, s1] += 1

        return matrix
