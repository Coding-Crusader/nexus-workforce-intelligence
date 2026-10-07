"""
NEXUS Candidate Database Module
Relational database entity management for candidates, multi-source evidence,
skill events, derived capabilities, evaluation runs, and counterfactual simulations.
Local-first SQLite implementation complying with SAS hackathon data security rules.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

DB_PATH = Path("data/nexus_candidates.db")

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection with foreign keys enabled and row factory set."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_database(db_path: Path = DB_PATH):
    """Creates the relational candidate schema if not already present."""
    conn = get_connection(db_path)
    with conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            current_title TEXT,
            years_of_experience REAL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS candidate_sources (
            source_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            source_type TEXT NOT NULL, -- RESUME, GITHUB, SKILL_PLATFORM, PORTFOLIO, SELF_DECLARED
            source_identifier TEXT NOT NULL,
            status TEXT DEFAULT 'ACTIVE',
            last_updated TEXT NOT NULL,
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS candidate_evidence (
            evidence_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            evidence_type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            timestamp TEXT NOT NULL,
            verification_strength TEXT NOT NULL, -- STRONG, MEDIUM, SUPPORTING, WEAK
            base_score REAL NOT NULL,
            recency_months REAL DEFAULT 1.0,
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            FOREIGN KEY (source_id) REFERENCES candidate_sources(source_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS candidate_skill_events (
            event_id TEXT PRIMARY KEY,
            evidence_id TEXT NOT NULL,
            skill TEXT NOT NULL,
            topic TEXT,
            score REAL NOT NULL,
            difficulty TEXT, -- EASY, MEDIUM, HARD
            volume INTEGER DEFAULT 1,
            recency REAL DEFAULT 1.0,
            metadata_json TEXT,
            FOREIGN KEY (evidence_id) REFERENCES candidate_evidence(evidence_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS candidate_skills (
            candidate_id TEXT NOT NULL,
            canonical_skill TEXT NOT NULL,
            capability_score REAL NOT NULL,
            confidence REAL NOT NULL,
            evidence_count INTEGER DEFAULT 0,
            last_evidence_date TEXT,
            evidence_summary_json TEXT,
            PRIMARY KEY (candidate_id, canonical_skill),
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS evaluation_runs (
            run_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            evaluated_at TEXT NOT NULL,
            reachable_count INTEGER NOT NULL,
            stretch_count INTEGER NOT NULL,
            blocked_count INTEGER NOT NULL,
            avg_fit_pct REAL NOT NULL,
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS role_evaluations (
            eval_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            role_name TEXT NOT NULL,
            compatibility_score REAL NOT NULL,
            status TEXT NOT NULL,
            benchmark_salary REAL,
            market_vacancies INTEGER,
            FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS simulation_runs (
            sim_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            intervention_id TEXT NOT NULL,
            simulated_at TEXT NOT NULL,
            reachable_before INTEGER NOT NULL,
            reachable_after INTEGER NOT NULL,
            expansion_pct REAL NOT NULL,
            salary_lift REAL NOT NULL,
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE
        );
        """)
    conn.close()

class CandidateDB:
    """Interface for managing candidate records, evidence ingestion, and derived skills."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        init_database(db_path)

    def get_or_create_candidate(
        self,
        email: str,
        name: Optional[str] = None,
        title: Optional[str] = None,
        years_exp: float = 2.0
    ) -> Dict[str, Any]:
        """Looks up candidate by email or creates a new candidate entity."""
        email = email.strip().lower()
        now_iso = datetime.now(timezone.utc).isoformat()
        
        conn = get_connection(self.db_path)
        with conn:
            cur = conn.execute("SELECT * FROM candidates WHERE email = ?", (email,))
            row = cur.fetchone()
            if row:
                candidate_id = row["candidate_id"]
                # Update name/title/exp if provided
                updates = []
                params = []
                if name and name != row["name"]:
                    updates.append("name = ?")
                    params.append(name)
                if title and title != row["current_title"]:
                    updates.append("current_title = ?")
                    params.append(title)
                if years_exp and years_exp != row["years_of_experience"]:
                    updates.append("years_of_experience = ?")
                    params.append(years_exp)
                
                if updates:
                    updates.append("updated_at = ?")
                    params.append(now_iso)
                    params.append(candidate_id)
                    conn.execute(f"UPDATE candidates SET {', '.join(updates)} WHERE candidate_id = ?", params)
            else:
                candidate_id = f"cand_{abs(hash(email)) % 10000000:07d}"
                conn.execute(
                    """INSERT INTO candidates 
                       (candidate_id, email, name, current_title, years_of_experience, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (candidate_id, email, name or "Candidate", title or "Analyst", years_exp, now_iso, now_iso)
                )

        # Retrieve fresh row
        with conn:
            cur = conn.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,))
            cand_row = dict(cur.fetchone())
        conn.close()
        return cand_row

    def register_source(
        self,
        candidate_id: str,
        source_type: str,
        source_identifier: str
    ) -> str:
        """Registers a data source (e.g. RESUME, GITHUB, SKILL_PLATFORM) for a candidate."""
        source_type = source_type.upper()
        source_id = f"src_{source_type[:3]}_{abs(hash(candidate_id + source_identifier)) % 1000000:06d}"
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = get_connection(self.db_path)
        with conn:
            conn.execute(
                """INSERT INTO candidate_sources (source_id, candidate_id, source_type, source_identifier, status, last_updated)
                   VALUES (?, ?, ?, ?, 'ACTIVE', ?)
                   ON CONFLICT(source_id) DO UPDATE SET last_updated = ?, status = 'ACTIVE'""",
                (source_id, candidate_id, source_type, source_identifier, now_iso, now_iso)
            )
        conn.close()
        return source_id

    def add_evidence(
        self,
        candidate_id: str,
        source_id: str,
        evidence_type: str,
        title: str,
        description: str,
        verification_strength: str,
        base_score: float,
        recency_months: float,
        skill_events: List[Dict[str, Any]]
    ) -> str:
        """Stores an auditable piece of evidence and associated skill demonstration events."""
        now_iso = datetime.now(timezone.utc).isoformat()
        evidence_id = f"evi_{abs(hash(candidate_id + title + now_iso)) % 10000000:07d}"

        conn = get_connection(self.db_path)
        with conn:
            conn.execute(
                """INSERT INTO candidate_evidence 
                   (evidence_id, candidate_id, source_id, evidence_type, title, description, timestamp, verification_strength, base_score, recency_months)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (evidence_id, candidate_id, source_id, evidence_type, title, description, now_iso, verification_strength.upper(), base_score, recency_months)
            )

            for idx, ev in enumerate(skill_events):
                event_id = f"{evidence_id}_ev{idx}"
                metadata_str = json.dumps(ev.get("metadata", {}))
                conn.execute(
                    """INSERT INTO candidate_skill_events 
                       (event_id, evidence_id, skill, topic, score, difficulty, volume, recency, metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event_id, evidence_id, ev["skill"], ev.get("topic", ""),
                        ev.get("score", base_score), ev.get("difficulty", "MEDIUM"),
                        ev.get("volume", 1), ev.get("recency", recency_months), metadata_str
                    )
                )
        conn.close()
        return evidence_id

    def get_candidate_sources(self, candidate_id: str) -> List[Dict[str, Any]]:
        """Returns all connected sources for a candidate."""
        conn = get_connection(self.db_path)
        with conn:
            cur = conn.execute("SELECT * FROM candidate_sources WHERE candidate_id = ?", (candidate_id,))
            rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def get_candidate_evidence_summary(self, candidate_id: str) -> List[Dict[str, Any]]:
        """Returns all evidence items joined with their source for auditable display."""
        conn = get_connection(self.db_path)
        with conn:
            cur = conn.execute("""
                SELECT e.*, s.source_type, s.source_identifier
                FROM candidate_evidence e
                JOIN candidate_sources s ON e.source_id = s.source_id
                WHERE e.candidate_id = ?
                ORDER BY e.timestamp DESC
            """, (candidate_id,))
            rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def save_derived_skills(
        self,
        candidate_id: str,
        skills_map: Dict[str, float],
        confidence_map: Dict[str, float],
        summaries_map: Dict[str, List[str]]
    ):
        """Saves derived canonical skill scores, confidence levels, and provenance summaries."""
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = get_connection(self.db_path)
        with conn:
            for skill_key, score in skills_map.items():
                conf = confidence_map.get(skill_key, 0.70)
                reasons = summaries_map.get(skill_key, [])
                conn.execute("""
                    INSERT INTO candidate_skills
                    (candidate_id, canonical_skill, capability_score, confidence, evidence_count, last_evidence_date, evidence_summary_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(candidate_id, canonical_skill) DO UPDATE SET
                        capability_score = excluded.capability_score,
                        confidence = excluded.confidence,
                        evidence_count = excluded.evidence_count,
                        last_evidence_date = excluded.last_evidence_date,
                        evidence_summary_json = excluded.evidence_summary_json
                """, (
                    candidate_id, skill_key, round(score, 3), round(conf, 3),
                    len(reasons), now_iso, json.dumps(reasons)
                ))
        conn.close()

    def get_derived_skills(self, candidate_id: str) -> Dict[str, Dict[str, Any]]:
        """Returns the derived skill profile with score, confidence, and explanation bullets."""
        conn = get_connection(self.db_path)
        with conn:
            cur = conn.execute("SELECT * FROM candidate_skills WHERE candidate_id = ?", (candidate_id,))
            rows = cur.fetchall()
            result = {}
            for r in rows:
                result[r["canonical_skill"]] = {
                    "score": float(r["capability_score"]),
                    "confidence": float(r["confidence"]),
                    "evidence_count": int(r["evidence_count"]),
                    "summary_bullets": json.loads(r["evidence_summary_json"] or "[]")
                }
        conn.close()
        return result

    def get_skill_events_for_candidate(self, candidate_id: str) -> List[Dict[str, Any]]:
        """Retrieves all skill events to feed into the scoring pipeline."""
        conn = get_connection(self.db_path)
        with conn:
            cur = conn.execute("""
                SELECT ev.*, e.verification_strength, e.recency_months, s.source_type
                FROM candidate_skill_events ev
                JOIN candidate_evidence e ON ev.evidence_id = e.evidence_id
                JOIN candidate_sources s ON e.source_id = s.source_id
                WHERE e.candidate_id = ?
            """, (candidate_id,))
            rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def seed_demo_candidate_if_missing(self) -> Dict[str, Any]:
        """Seeds Aarav Sharma into the relational database if not present."""
        email = "aarav.sharma@example.com"
        cand = self.get_or_create_candidate(
            email=email,
            name="Aarav Sharma",
            title="Associate Data Analyst",
            years_exp=2.0
        )
        sources = self.get_candidate_sources(cand["candidate_id"])
        if not sources:
            from backend.data_loader import get_demo_candidate
            demo_json = get_demo_candidate()
            portfolio = demo_json.get("evidence_portfolio", [])
            
            src_map = {
                "RESUME": self.register_source(cand["candidate_id"], "RESUME", "Aarav_Sharma_Resume.pdf"),
                "GITHUB": self.register_source(cand["candidate_id"], "GITHUB", "github.com/aaravsharma"),
                "SKILL_PLATFORM": self.register_source(cand["candidate_id"], "SKILL_PLATFORM", "HackerRank_Verified"),
                "PORTFOLIO": self.register_source(cand["candidate_id"], "PORTFOLIO", "Analytics_Project_Portfolio")
            }

            for item in portfolio:
                s_type = "RESUME"
                st_lower = item.get("source_type", "").lower()
                if "github" in st_lower:
                    s_type = "GITHUB"
                elif "assessment" in st_lower:
                    s_type = "SKILL_PLATFORM"
                elif "project" in st_lower:
                    s_type = "PORTFOLIO"
                
                src_id = src_map.get(s_type, src_map["RESUME"])
                skill_k = item.get("skill_key", "python")
                if skill_k == "sql_database":
                    skill_k = "sql"
                elif skill_k == "math_statistics":
                    skill_k = "statistics_math"
                elif skill_k == "visualization_storytelling":
                    skill_k = "business_intelligence"

                self.add_evidence(
                    candidate_id=cand["candidate_id"],
                    source_id=src_id,
                    evidence_type="PORTFOLIO_VERIFIED",
                    title=f"Verified Evidence: {item.get('skill_name')}",
                    description=item.get("evidence_description", ""),
                    verification_strength=item.get("evidence_strength", "STRONG"),
                    base_score=float(item.get("base_score", 0.80)),
                    recency_months=float(item.get("recency_months", 2.0)),
                    skill_events=[{
                        "skill": skill_k,
                        "topic": item.get("skill_name"),
                        "score": float(item.get("base_score", 0.80)),
                        "difficulty": "MEDIUM",
                        "volume": 1,
                        "recency": float(item.get("recency_months", 2.0))
                    }]
                )
        return cand

