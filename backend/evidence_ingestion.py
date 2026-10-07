"""
NEXUS Evidence Ingestion & Source Connectors
Connectors for Resume (PDF/DOCX), GitHub public repositories,
Skill-Platform exports (LeetCode/HackerRank/Kaggle CSV/JSON), and Portfolio projects.
Follows competition-safe local extraction without unauthorized scraping.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import csv
import io
import re
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from backend.data_loader import get_skill_aliases
from backend.resume_parser import extract_capabilities_from_resume

class ResumeConnector:
    """Extracts verified capability evidence from uploaded resumes (PDF / DOCX)."""

    def __init__(self):
        self.aliases = get_skill_aliases()

    def process(self, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        extracted = extract_capabilities_from_resume(file_bytes=file_bytes, filename=filename)
        caps = extracted.get("capabilities", {})
        cand_name = extracted.get("name", "Candidate")
        exp_years = extracted.get("years_of_experience", 2.0)

        skill_counts = extracted.get("raw_mention_counts", {})
        skill_events = []
        for skill_key, score in caps.items():
            cnt = skill_counts.get(skill_key, 0)
            if cnt > 0 and score > 0.25:
                skill_events.append({
                    "skill": skill_key,
                    "topic": f"Resume Documentation ({cnt} mention{'s' if cnt != 1 else ''})",
                    "score": round(score, 3),
                    "difficulty": "HARD" if cnt >= 5 else ("MEDIUM" if cnt >= 2 else "EASY"),
                    "volume": cnt,
                    "recency": 2.0,
                    "metadata": {"source_doc": filename, "mentions": cnt}
                })

        # Slide 04: Unverified resume text claims without live repository verification are SUPPORTING/WEAK
        verif_strength = "MEDIUM" if exp_years >= 3.0 else ("SUPPORTING" if exp_years >= 1.0 else "WEAK")
        base_score = 0.80 if exp_years >= 3.0 else 0.70

        return {
            "source_type": "RESUME",
            "source_identifier": filename,
            "candidate_name": cand_name,
            "experience_years": exp_years,
            "evidence_type": "RESUME_VERIFICATION",
            "title": f"Resume Documentation: {filename}",
            "description": f"Extracted professional experience and skills from {filename}.",
            "verification_strength": verif_strength,
            "base_score": base_score,
            "recency_months": 2.0,
            "skill_events": skill_events
        }

class GitHubConnector:
    """
    Connects to public GitHub profile metadata or local JSON export.
    Never bypasses authentication or scrapes private repositories.
    """

    def __init__(self):
        self.aliases = get_skill_aliases()

    def process_export_data(self, data: List[Dict[str, Any]], username: str = "user") -> Dict[str, Any]:
        """
        Processes repository metadata (from public API or uploaded JSON export).
        Expected repo keys: name, language, description, topics, stargazers_count, updated_at.
        """
        lang_counts = {}
        topic_counts = {}
        total_repos = len(data)

        for repo in data:
            lang = str(repo.get("language") or "").strip().lower()
            if lang and lang != "none":
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

            topics = repo.get("topics", [])
            for t in topics:
                t_clean = str(t).strip().lower()
                topic_counts[t_clean] = topic_counts.get(t_clean, 0) + 1

            desc = str(repo.get("description") or "").lower()
            for skill_key, kws in self.aliases.items():
                keywords = kws if isinstance(kws, list) else kws.get("keywords", [])
                for kw in keywords:
                    if kw in desc or kw in topics:
                        topic_counts[skill_key] = topic_counts.get(skill_key, 0) + 1

        skill_events = []
        # Map languages & topics to canonical skills
        lang_mapping = {
            "python": "python",
            "jupyter notebook": "python",
            "c++": "c_plus_plus",
            "cpp": "c_plus_plus",
            "c": "c_plus_plus",
            "sql": "sql",
            "dockerfile": "mlops_production",
            "shell": "data_engineering"
        }

        for lang, count in lang_counts.items():
            canonical = lang_mapping.get(lang)
            if canonical:
                skill_events.append({
                    "skill": canonical,
                    "topic": f"GitHub Repositories in {lang.title()}",
                    "score": min(0.95, 0.50 + (count * 0.08)),
                    "difficulty": "HARD" if count >= 5 else "MEDIUM",
                    "volume": count,
                    "recency": 1.5,
                    "metadata": {"repo_count": count}
                })

        for topic, count in topic_counts.items():
            if topic in self.aliases:
                skill_events.append({
                    "skill": topic,
                    "topic": f"GitHub Repositories tagged #{topic}",
                    "score": min(0.90, 0.45 + (count * 0.10)),
                    "difficulty": "MEDIUM",
                    "volume": count,
                    "recency": 2.0,
                    "metadata": {"tagged_projects": count}
                })

        return {
            "source_type": "GITHUB",
            "source_identifier": f"github.com/{username}",
            "evidence_type": "PUBLIC_CODE_REPOSITORIES",
            "title": f"GitHub Public Code Artifacts (@{username})",
            "description": f"Audited {total_repos} public repositories, languages ({list(lang_counts.keys())}), and engineering topics.",
            "verification_strength": "STRONG",
            "base_score": 0.88,
            "recency_months": 1.5,
            "skill_events": skill_events
        }

    def search_users_by_email(self, email: str) -> List[Dict[str, Any]]:
        """Searches GitHub public users API using email or email handle to find matched profiles."""
        import urllib.request
        import urllib.parse
        headers = {"User-Agent": "NEXUS-Evidence-Connector/1.0"}
        matches = []
        clean_email = email.strip()
        prefix = clean_email.split("@")[0] if "@" in clean_email else clean_email
        
        # 1. Search by email
        query = urllib.parse.quote(f"{clean_email} in:email")
        url = f"https://api.github.com/search/users?q={query}&per_page=5"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for item in data.get("items", []):
                    matches.append({
                        "login": item.get("login"),
                        "avatar_url": item.get("avatar_url", ""),
                        "html_url": item.get("html_url", ""),
                        "match_type": "Email match"
                    })
        except Exception:
            pass

        # 2. Search by handle prefix if needed
        if len(matches) < 3 and prefix:
            query_prefix = urllib.parse.quote(prefix)
            url_prefix = f"https://api.github.com/search/users?q={query_prefix}&per_page=5"
            try:
                req = urllib.request.Request(url_prefix, headers=headers)
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    existing_logins = {m["login"].lower() for m in matches}
                    for item in data.get("items", []):
                        login = item.get("login")
                        if login and login.lower() not in existing_logins:
                            matches.append({
                                "login": login,
                                "avatar_url": item.get("avatar_url", ""),
                                "html_url": item.get("html_url", ""),
                                "match_type": "Username match"
                            })
            except Exception:
                pass

        return matches[:6]

    def fetch_public_profile(self, username: str) -> Dict[str, Any]:
        """Attempts to read public repositories AND user profile from GitHub public API without auth.
        Returns both evidence dict and 'user_profile' key with name, bio, avatar_url.
        Never injects mock or filler repositories if user has none.
        """
        import urllib.request
        username = username.strip().replace("@", "")
        headers = {"User-Agent": "NEXUS-Evidence-Connector/1.0"}

        # Fetch user profile info (name, bio, public repos count)
        user_profile = {"login": username, "name": username, "bio": "", "avatar_url": "", "public_repos": 0}
        try:
            profile_req = urllib.request.Request(
                f"https://api.github.com/users/{username}", headers=headers
            )
            with urllib.request.urlopen(profile_req, timeout=4.0) as resp:
                profile_data = json.loads(resp.read().decode("utf-8"))
                user_profile = {
                    "login": profile_data.get("login", username),
                    "name": profile_data.get("name") or username,
                    "bio": profile_data.get("bio") or "",
                    "avatar_url": profile_data.get("avatar_url", ""),
                    "public_repos": profile_data.get("public_repos", 0),
                    "company": profile_data.get("company") or "",
                    "location": profile_data.get("location") or "",
                }
        except Exception:
            pass

        # Fetch repositories
        url = f"https://api.github.com/users/{username}/repos?per_page=30&sort=updated"
        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list) and len(data) > 0:
                    evidence = self.process_export_data(data, username=username)
                else:
                    evidence = {
                        "source_type": "GITHUB",
                        "source_identifier": f"github.com/{username}",
                        "evidence_type": "PUBLIC_CODE_REPOSITORIES",
                        "title": f"GitHub Profile (@{username})",
                        "description": f"Audited public profile @{username}: 0 public repositories found.",
                        "verification_strength": "LOW",
                        "base_score": 0.50,
                        "recency_months": 1.0,
                        "skill_events": []
                    }
        except Exception:
            evidence = {
                "source_type": "GITHUB",
                "source_identifier": f"github.com/{username}",
                "evidence_type": "PUBLIC_CODE_REPOSITORIES",
                "title": f"GitHub Profile (@{username})",
                "description": f"No public repositories accessible for @{username}.",
                "verification_strength": "LOW",
                "base_score": 0.50,
                "recency_months": 1.0,
                "skill_events": []
            }

        evidence["user_profile"] = user_profile
        return evidence


class LeetCodeConnector:
    """
    Fetches publicly available LeetCode problem-solving statistics using the
    public GraphQL endpoint. No authentication or login required.
    Returns Easy/Medium/Hard accepted counts and derives capability evidence.
    """

    GRAPHQL_URL = "https://leetcode.com/graphql"
    QUERY = """
    query getUserProfile($username: String!) {
      matchedUser(username: $username) {
        username
        submitStatsGlobal {
          acSubmissionNum {
            difficulty
            count
          }
        }
      }
    }
    """

    def fetch_public_stats(self, username: str) -> Dict[str, Any]:
        """
        Calls LeetCode public GraphQL API to retrieve accepted submission counts.
        Returns structured evidence record with skill_events derived from difficulty distribution.
        Falls back gracefully if network is unavailable.
        """
        import urllib.request
        username = username.strip()

        counts = {"All": 0, "Easy": 0, "Medium": 0, "Hard": 0}

        try:
            payload = json.dumps({
                "query": self.QUERY,
                "variables": {"username": username}
            }).encode("utf-8")

            req = urllib.request.Request(
                self.GRAPHQL_URL,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "NEXUS-Evidence-Connector/1.0",
                    "Referer": "https://leetcode.com"
                },
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=5.0) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                user_data = result.get("data", {}).get("matchedUser")
                if user_data:
                    ac_list = user_data.get("submitStatsGlobal", {}).get("acSubmissionNum", [])
                    for item in ac_list:
                        diff = item.get("difficulty", "")
                        cnt = item.get("count", 0)
                        if diff in counts:
                            counts[diff] = cnt
        except Exception:
            # Network unavailable or user not found — return zero evidence gracefully
            pass

        total = counts.get("All", 0)
        easy = counts.get("Easy", 0)
        medium = counts.get("Medium", 0)
        hard = counts.get("Hard", 0)

        skill_events = []

        if total > 0:
            # Weighted difficulty score
            weighted_solved = easy * 0.60 + medium * 0.85 + hard * 1.0
            vol_factor = min(1.0, weighted_solved / 280.0)
            # Hard ratio bonus
            hard_ratio = hard / max(total, 1)
            difficulty_bonus = hard_ratio * 0.10

            base_score = min(0.97, 0.45 + (vol_factor * 0.40) + difficulty_bonus)

            # LeetCode heavy usage → algorithmic (python/c++ proxy) + sql (if any sql solved)
            skill_events.append({
                "skill": "python",
                "topic": f"LeetCode: {total} Accepted ({easy}E / {medium}M / {hard}H)",
                "score": round(base_score, 3),
                "difficulty": "HARD" if hard > 20 else "MEDIUM",
                "volume": total,
                "recency": 1.0,
                "metadata": {
                    "platform": "LeetCode",
                    "username": username,
                    "easy": easy,
                    "medium": medium,
                    "hard": hard,
                    "total_accepted": total
                }
            })

            # SQL hint from total volume (LeetCode DB problems exist in all plans)
            if total >= 50:
                sql_score = min(0.88, 0.50 + (min(total, 200) / 200) * 0.30)
                skill_events.append({
                    "skill": "sql",
                    "topic": f"LeetCode Database Problems (inferred from {total} solved)",
                    "score": round(sql_score, 3),
                    "difficulty": "MEDIUM",
                    "volume": max(1, total // 10),
                    "recency": 1.0,
                    "metadata": {"platform": "LeetCode", "inferred": True}
                })

            # Hard problems → ML/stats signal
            if hard >= 15:
                ml_score = min(0.82, 0.50 + (hard / 100) * 0.30)
                skill_events.append({
                    "skill": "statistics_math",
                    "topic": f"LeetCode Hard Problems ({hard} solved) → Algorithmic Reasoning",
                    "score": round(ml_score, 3),
                    "difficulty": "HARD",
                    "volume": hard,
                    "recency": 1.0,
                    "metadata": {"platform": "LeetCode", "hard_problems": hard}
                })

        return {
            "source_type": "SKILL_PLATFORM",
            "source_identifier": f"leetcode.com/{username}",
            "evidence_type": "STANDARDIZED_PROBLEM_ACTIVITY",
            "title": f"LeetCode Public Profile (@{username}) — {total} Accepted",
            "description": (
                f"Fetched {total} accepted solutions ({easy} Easy / {medium} Medium / {hard} Hard) "
                f"from LeetCode public GraphQL API for user '{username}'."
            ) if total > 0 else f"LeetCode profile '{username}' fetched but no accepted solutions found or user not found.",
            "verification_strength": "STRONG" if total >= 100 else ("MEDIUM" if total >= 20 else "WEAK"),
            "base_score": 0.90,
            "recency_months": 1.0,
            "skill_events": skill_events,
            "raw_counts": counts
        }


def infer_usernames_from_email(email: str, name: str = "") -> Dict[str, List[str]]:
    """
    Infers likely GitHub and LeetCode usernames from an email address and optional name.
    Returns a dict with "github" and "leetcode" keys containing candidate username strings to try.
    All inference is local — no network calls made here.

    Inference strategy:
    1. email prefix (before @), stripped of numbers → primary candidate
    2. firstname from name (lowercase, no spaces)
    3. firstname.lastname from name
    4. email prefix with dots/underscores preserved
    """
    candidates_github: List[str] = []
    candidates_leetcode: List[str] = []

    # From email prefix
    if email and "@" in email:
        prefix = email.split("@")[0]
        # Remove trailing digits (e.g. aryan123 → aryan)
        prefix_stripped = re.sub(r"\d+$", "", prefix)
        if prefix_stripped and prefix_stripped != prefix:
            candidates_github.insert(0, prefix_stripped)
            candidates_leetcode.insert(0, prefix_stripped)
        candidates_github.append(prefix)
        candidates_leetcode.append(prefix)

    # From display name
    if name and name.strip():
        parts = name.strip().lower().split()
        if parts:
            first = re.sub(r"[^a-z0-9]", "", parts[0])
            if first:
                candidates_github.append(first)
                candidates_leetcode.append(first)
            if len(parts) >= 2:
                last = re.sub(r"[^a-z0-9]", "", parts[-1])
                if first and last:
                    combined = f"{first}{last}"
                    dotted = f"{first}.{last}"
                    dashed = f"{first}-{last}"
                    candidates_github += [combined, dotted, dashed]
                    candidates_leetcode += [combined, dotted]

    # Deduplicate while preserving order
    seen: set = set()
    unique_gh = []
    for u in candidates_github:
        if u and u not in seen:
            seen.add(u)
            unique_gh.append(u)

    seen2: set = set()
    unique_lc = []
    for u in candidates_leetcode:
        if u and u not in seen2:
            seen2.add(u)
            unique_lc.append(u)

    return {"github": unique_gh[:5], "leetcode": unique_lc[:5]}

class SkillPlatformConnector:
    """
    Interprets skill-platform activity exports (LeetCode, HackerRank, Codeforces, Kaggle).
    Parses problems solved, language used, topic, difficulty, success rate, and timestamps.
    """

    def __init__(self):
        self.aliases = get_skill_aliases()

    def process_csv_or_df(self, df: pd.DataFrame, platform_name: str = "SkillPlatform") -> Dict[str, Any]:
        """
        Parses problem-solving records.
        Flexible column matching:
        - Language: language, lang, programming_language
        - Topic: topic, tag, category, concept
        - Difficulty: difficulty, level
        - Result: result, status, verdict, outcome (Accepted/Solved)
        - Date: date, timestamp, solved_at
        """
        # Normalize column names
        cols_lower = {c: str(c).strip().lower() for c in df.columns}
        inv_cols = {v: k for k, v in cols_lower.items()}

        col_lang = inv_cols.get("language") or inv_cols.get("lang") or inv_cols.get("programming_language")
        col_topic = inv_cols.get("topic") or inv_cols.get("tag") or inv_cols.get("category")
        col_diff = inv_cols.get("difficulty") or inv_cols.get("level")
        col_status = inv_cols.get("result") or inv_cols.get("status") or inv_cols.get("verdict")
        col_date = inv_cols.get("date") or inv_cols.get("timestamp") or inv_cols.get("solved_at")

        total_records = len(df)
        
        # Filter for successful/accepted problems if status column is present
        if col_status:
            success_mask = df[col_status].astype(str).str.lower().str.contains("accept|solve|pass|success|100")
            df_success = df[success_mask].copy()
        else:
            df_success = df.copy()

        success_count = len(df_success)
        success_rate = (success_count / total_records) if total_records > 0 else 1.0

        # Group by language
        skill_events = []
        lang_mapping = {
            "c++": "c_plus_plus",
            "cpp": "c_plus_plus",
            "python": "python",
            "python3": "python",
            "sql": "sql",
            "java": "data_engineering",
            "r": "statistics_math"
        }

        if col_lang and col_lang in df_success.columns:
            for lang_val, group in df_success.groupby(col_lang):
                lang_str = str(lang_val).strip().lower()
                canonical = lang_mapping.get(lang_str, lang_str)
                count = len(group)
                
                # Check difficulty distribution
                diff_weights = {"easy": 0.6, "medium": 0.85, "hard": 1.0}
                if col_diff and col_diff in group.columns:
                    diff_series = group[col_diff].astype(str).str.lower()
                    avg_diff_weight = diff_series.map(lambda d: diff_weights.get(d, 0.75)).mean()
                    hard_count = int(diff_series.str.contains("hard").sum())
                    med_count = int(diff_series.str.contains("medium|med").sum())
                else:
                    avg_diff_weight = 0.75
                    hard_count = 0
                    med_count = int(count * 0.6)

                # Score derived from volume, difficulty, and success rate
                # e.g., 400+ problems with 80%+ success and medium/hard distribution gives 0.85-0.95
                vol_factor = min(1.0, count / 300.0)
                raw_score = (0.50 + (0.35 * vol_factor) + (0.15 * (avg_diff_weight - 0.5) / 0.5)) * min(1.0, success_rate + 0.15)
                final_score = min(0.98, max(0.40, raw_score))

                skill_events.append({
                    "skill": canonical,
                    "topic": f"Algorithmic Problem Solving ({lang_str.upper()})",
                    "score": round(final_score, 3),
                    "difficulty": "HARD" if hard_count > 10 else "MEDIUM",
                    "volume": count,
                    "recency": 1.0,
                    "metadata": {
                        "problems_solved": count,
                        "success_rate_pct": round(success_rate * 100, 1),
                        "medium_problems": med_count,
                        "hard_problems": hard_count
                    }
                })

        # Also group by topic (e.g. Dynamic Programming, Trees, Database, Regression)
        if col_topic and col_topic in df_success.columns:
            for topic_val, group in df_success.groupby(col_topic):
                t_str = str(topic_val).strip().lower()
                count = len(group)
                if count >= 5:
                    # Check if maps to canonical skills
                    matched_canonical = None
                    if any(w in t_str for w in ["sql", "database", "query"]):
                        matched_canonical = "sql"
                    elif any(w in t_str for w in ["dynamic programming", "graph", "tree", "algorithm"]):
                        matched_canonical = "c_plus_plus"  # algorithmic problem solving
                    elif any(w in t_str for w in ["machine learning", "ml", "classification"]):
                        matched_canonical = "machine_learning"

                    if matched_canonical:
                        skill_events.append({
                            "skill": matched_canonical,
                            "topic": f"Topic Mastery: {topic_val}",
                            "score": min(0.92, 0.55 + (count * 0.02)),
                            "difficulty": "MEDIUM",
                            "volume": count,
                            "recency": 1.0,
                            "metadata": {"topic_problems": count}
                        })

        return {
            "source_type": "SKILL_PLATFORM",
            "source_identifier": platform_name,
            "evidence_type": "STANDARDIZED_PROBLEM_ACTIVITY",
            "title": f"{platform_name} Activity Export ({success_count:,} Problems Solved)",
            "description": f"Verified {success_count} solved problems ({success_rate*100:.0f}% success rate) across {len(skill_events)} technology domains.",
            "verification_strength": "STRONG",
            "base_score": 0.90,
            "recency_months": 1.0,
            "skill_events": skill_events
        }
