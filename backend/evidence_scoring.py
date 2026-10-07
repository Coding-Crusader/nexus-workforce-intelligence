"""
NEXUS Evidence-to-Capability Scoring Engine
Aggregates multi-source evidence into canonical skill capabilities with
exponential recency half-life decay, verification strength weighting,
multi-source cross-validation bonuses, and human-auditable "WHY?" explanations.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import json
from collections import defaultdict

from backend.data_loader import get_skill_aliases

STRENGTH_WEIGHTS = {
    "STRONG": 1.00,
    "MEDIUM": 0.85,
    "SUPPORTING": 0.75,
    "WEAK": 0.60,
    "SELF_DECLARED": 0.50
}

CANONICAL_KEYS = [
    "python", "sql", "machine_learning", "statistics_math",
    "business_intelligence", "data_storytelling", "big_data_cloud",
    "deep_learning_nlp", "data_engineering", "mlops_production"
]

SKILL_NORMALIZATION_MAP = {
    # database & sql
    "sql": "sql",
    "sql_database": "sql",
    "mysql": "sql",
    "postgresql": "sql",
    "plsql": "sql",
    "pl/sql": "sql",
    "database": "sql",
    "rdbms": "sql",
    # math & stats
    "statistics_math": "statistics_math",
    "math_statistics": "statistics_math",
    "maths_stats": "statistics_math",
    "maths-stats_skills": "statistics_math",
    "statistics": "statistics_math",
    # machine learning & deep learning
    "machine_learning": "machine_learning",
    "ai_and_ml_skills": "machine_learning",
    "deep_learning_nlp": "deep_learning_nlp",
    "deep_learning": "deep_learning_nlp",
    "nlp": "deep_learning_nlp",
    # business analysis & BI
    "business_intelligence": "business_intelligence",
    "business_analysis": "business_intelligence",
    "sas": "business_intelligence",
    # visualization & storytelling
    "data_storytelling": "data_storytelling",
    "visualization_storytelling": "data_storytelling",
    "dashboard_and_storytelling_skills": "data_storytelling",
    "power bi": "data_storytelling",
    "tableau": "data_storytelling",
    # big data & engineering
    "big_data_cloud": "big_data_cloud",
    "big_data_engineering": "big_data_cloud",
    "big_data_skills": "big_data_cloud",
    "big data": "big_data_cloud",
    "data_engineering": "data_engineering",
    # mlops & cloud
    "mlops_production": "mlops_production",
    "cloud_deployment_mlops": "mlops_production",
    # programming & algorithms
    "python": "python",
    "coding_skills": "python",
    "c_plus_plus": "python",
    "cpp": "python",
    "c": "python",
    "java": "python",
    "r_programming": "statistics_math",
    "algorithms": "python",
    "data_structures": "python"
}

class EvidenceScorer:
    """Computes derived canonical capabilities from multi-source evidence events."""

    def __init__(self, half_life_months: float = 18.0):
        self.half_life_months = half_life_months
        self.aliases = get_skill_aliases()

    def score_candidate_events(
        self,
        skill_events: List[Dict[str, Any]],
        existing_profile: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Processes all skill events across all connected sources:
        1. Groups events by canonical skill.
        2. Applies verification strength multipliers and recency half-life decay.
        3. Awards multi-source cross-validation bonus when 2+ distinct sources corroborate.
        4. Derives final capability score [0.0, 1.0].
        5. Formulates human-auditable 'WHY?' explanation points.
        """
        # Group events by canonical skill
        skill_groups = defaultdict(list)
        for ev in skill_events:
            skill = str(ev.get("skill", "")).lower().strip()
            # 1. Direct normalization map lookup
            if skill in SKILL_NORMALIZATION_MAP:
                canonical_target = SKILL_NORMALIZATION_MAP[skill]
            elif skill in CANONICAL_KEYS:
                canonical_target = skill
            else:
                # 2. Search alias dictionary
                canonical_target = "python"
                for c_k, kws in self.aliases.items():
                    keywords = kws if isinstance(kws, list) else kws.get("keywords", [])
                    if skill in keywords or any(k in skill for k in keywords):
                        canonical_target = SKILL_NORMALIZATION_MAP.get(c_k, "python")
                        break

            skill_groups[canonical_target].append(ev)

        derived_scores = {}
        confidence_map = {}
        explanations_map = {}

        for skill_key in CANONICAL_KEYS:
            events = skill_groups.get(skill_key, [])
            if not events:
                # If existing profile has a baseline, preserve or default to entry-level
                baseline = existing_profile.get(skill_key, 0.20) if existing_profile else 0.20
                derived_scores[skill_key] = round(baseline, 3)
                confidence_map[skill_key] = 0.50
                explanations_map[skill_key] = [
                    "No direct external verified evidence ingested yet.",
                    "Estimated from foundational candidate background profile."
                ]
                continue

            sources_seen = set()
            weighted_scores = []
            weights = []
            bullet_reasons = []

            total_volume = 0
            has_hard = False
            best_recency = 999.0

            for ev in events:
                src_type = ev.get("source_type", "RESUME")
                sources_seen.add(src_type)
                strength_label = ev.get("verification_strength", "MEDIUM").upper()
                strength_mult = STRENGTH_WEIGHTS.get(strength_label, 0.75)

                recency = float(ev.get("recency_months", ev.get("recency", 1.0)))
                if recency < best_recency:
                    best_recency = recency

                decay = float(2.0 ** (-max(0.0, recency) / self.half_life_months))
                decay = max(0.25, min(1.0, decay))

                event_score = float(ev.get("score", 0.50))
                event_vol = int(ev.get("volume", 1))
                total_volume += event_vol

                if str(ev.get("difficulty", "")).upper() == "HARD":
                    has_hard = True

                effective_weight = strength_mult * decay * math.sqrt(max(1, event_vol))
                weighted_scores.append(event_score * effective_weight)
                weights.append(effective_weight)

            # Combined score
            if sum(weights) > 0:
                raw_combined = sum(weighted_scores) / sum(weights)
            else:
                raw_combined = 0.50

            # Multi-source cross-verification bonus: +5% if verified across 2+ independent sources
            source_bonus = 0.05 if len(sources_seen) >= 2 else 0.0
            volume_bonus = min(0.10, math.log10(max(1, total_volume)) * 0.04)

            final_score = min(0.98, max(0.15, raw_combined + source_bonus + volume_bonus))
            derived_scores[skill_key] = round(final_score, 3)

            # Confidence calculation
            confidence = min(0.95, 0.60 + (0.15 if len(sources_seen) >= 2 else 0.0) + (0.10 if best_recency <= 3.0 else 0.05))
            confidence_map[skill_key] = round(confidence, 3)

            # Formulate human-auditable "WHY?" explanations
            bullet_reasons.append(f"Derived from {len(events)} verified evidence demonstrations across {len(sources_seen)} independent source(s): {', '.join(sorted(sources_seen))}.")
            if total_volume > 1:
                bullet_reasons.append(f"Demonstrated across {total_volume:,} distinct verified activity items or projects.")
            if best_recency <= 3.0:
                bullet_reasons.append(f"High evidence recency (active demonstration within the last {best_recency:.1f} months).")
            if has_hard:
                bullet_reasons.append("Advanced/Hard difficulty problems or production-grade tasks successfully validated.")
            if len(sources_seen) >= 2:
                bullet_reasons.append(f"Cross-verified by multiple independent sources ({' & '.join(sorted(sources_seen))}).")

            explanations_map[skill_key] = bullet_reasons

        return {
            "capabilities": derived_scores,
            "confidence_scores": confidence_map,
            "provenance_explanations": explanations_map
        }
