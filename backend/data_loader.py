"""
NEXUS Data Loader Module
Locates, reads, standardizes, and caches the raw hackathon datasets and configuration files.
Prioritizes the official hackathon directory at:
C:/Coding_Crusader/NEXUS/SAS Data Problem Statement and Instructions Hackathon
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Resolve project root dynamically
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Primary and fallback data directories
PRIMARY_SAS_DIR = Path("C:/Coding_Crusader/NEXUS/SAS Data Problem Statement and Instructions Hackathon")

CANDIDATE_DATA_DIRS = [
    PRIMARY_SAS_DIR,
    PROJECT_ROOT / "SAS Data Problem Statement and Instructions Hackathon",
    PROJECT_ROOT / "data" / "raw",
    Path("C:/Coding_Crusader/NEXUS/data/raw"),
]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def find_file(filename: str) -> Path:
    """Searches prioritized data folders for the specified filename."""
    for folder in CANDIDATE_DATA_DIRS:
        candidate = folder / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Dataset '{filename}' could not be located in any known directories: {CANDIDATE_DATA_DIRS}")

def load_json(filepath: Path) -> Any:
    """Loads a JSON file with utf-8 encoding."""
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def get_settings() -> Dict[str, Any]:
    """Loads system settings from config/settings.json."""
    settings_path = PROJECT_ROOT / "config" / "settings.json"
    return load_json(settings_path)

def get_skill_aliases() -> Dict[str, list]:
    """Loads skill aliases mapping from config/skill_aliases.json."""
    aliases_path = PROJECT_ROOT / "config" / "skill_aliases.json"
    return load_json(aliases_path)

def get_interventions() -> list:
    """Loads defined transition interventions from config/interventions.json."""
    interventions_path = PROJECT_ROOT / "config" / "interventions.json"
    return load_json(interventions_path)

def get_demo_candidate() -> Dict[str, Any]:
    """Loads demo candidate profile from data/demo_candidate.json."""
    candidate_path = PROJECT_ROOT / "data" / "demo_candidate.json"
    return load_json(candidate_path)

def _load_raw_analytics_jobs() -> pd.DataFrame:
    """Loads Analytics Jobs.csv directly from primary hackathon folder."""
    fpath = find_file("Analytics Jobs.csv")
    df = pd.read_csv(fpath)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _load_raw_datascience_jobs() -> pd.DataFrame:
    """Loads DataScience Jobs.csv directly from primary hackathon folder."""
    fpath = find_file("DataScience Jobs.csv")
    df = pd.read_csv(fpath)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _load_raw_jds_skills() -> pd.DataFrame:
    """Loads JDS Skill Traits.xlsx directly from primary hackathon folder."""
    fpath = find_file("JDS Skill Traits.xlsx")
    df = pd.read_excel(fpath)
    df.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]
    return df

def _load_raw_sds_personality() -> pd.DataFrame:
    """Loads SDS Personality Traits.xlsx directly from primary hackathon folder."""
    fpath = find_file("SDS Personality Traits.xlsx")
    df = pd.read_excel(fpath)
    clean_cols = []
    for c in df.columns:
        c_clean = c.strip().lower().replace(" ", "_").replace("__", "_")
        if "success" in c_clean:
            c_clean = "success_classification_high_low"
        clean_cols.append(c_clean)
    df.columns = clean_cols
    return df

if HAS_STREAMLIT:
    @st.cache_data(show_spinner=False)
    def load_analytics_jobs() -> pd.DataFrame:
        return _load_raw_analytics_jobs()

    @st.cache_data(show_spinner=False)
    def load_datascience_jobs() -> pd.DataFrame:
        return _load_raw_datascience_jobs()

    @st.cache_data(show_spinner=False)
    def load_jds_skills() -> pd.DataFrame:
        return _load_raw_jds_skills()

    @st.cache_data(show_spinner=False)
    def load_sds_personality() -> pd.DataFrame:
        return _load_raw_sds_personality()
else:
    def load_analytics_jobs() -> pd.DataFrame:
        return _load_raw_analytics_jobs()

    def load_datascience_jobs() -> pd.DataFrame:
        return _load_raw_datascience_jobs()

    def load_jds_skills() -> pd.DataFrame:
        return _load_raw_jds_skills()

    def load_sds_personality() -> pd.DataFrame:
        return _load_raw_sds_personality()

def load_all_raw_datasets() -> Dict[str, pd.DataFrame]:
    """Loads all 4 primary datasets into a dictionary."""
    return {
        "analytics_jobs": load_analytics_jobs(),
        "datascience_jobs": load_datascience_jobs(),
        "jds_skills": load_jds_skills(),
        "sds_personality": load_sds_personality()
    }
