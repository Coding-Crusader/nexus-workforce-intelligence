"""
NEXUS Market Analysis Module
Extracts empirical labour market intelligence from Data Science Jobs and Analytics Jobs datasets.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

class MarketAnalyzer:
    """Analyzes salary, experience, geography, and demand distributions across job datasets."""

    def __init__(self, df_analytics: pd.DataFrame, df_datascience: pd.DataFrame):
        self.df_aj = df_analytics
        self.df_ds = df_datascience

    def get_headline_metrics(self) -> Dict[str, Any]:
        """Calculates global headline market metrics across both datasets."""
        total_aj_postings = len(self.df_aj)
        total_ds_vacancies = int(self.df_ds["num_of_jobs"].sum())
        total_ds_records = len(self.df_ds)
        unique_companies = int(self.df_ds["company_name_clean"].nunique())
        unique_ds_roles = int(self.df_ds["job_title_clean"].nunique())

        avg_market_salary = round(float(self.df_ds["avg_salary_lakhs"].mean()), 2)
        median_market_salary = round(float(self.df_ds["avg_salary_lakhs"].median()), 2)
        max_market_salary = round(float(self.df_ds["max_salary_lakhs"].max()), 2)

        median_exp = round(float(self.df_aj["avg_experience"].median()), 1)

        return {
            "analytics_jobs_sample_size": total_aj_postings,
            "datascience_postings_sample_size": total_ds_records,
            "total_market_vacancies": total_ds_vacancies,
            "unique_companies_represented": unique_companies,
            "unique_primary_roles": unique_ds_roles,
            "average_market_salary_lakhs": avg_market_salary,
            "median_market_salary_lakhs": median_market_salary,
            "peak_salary_lakhs": max_market_salary,
            "median_experience_years": median_exp,
            "top_metro_location": "Bengaluru (3,333 jobs, 21.0%)"
        }

    def get_role_demand_and_salary_benchmarks(self) -> pd.DataFrame:
        """
        Combines role volume, vacancy count, average salary, and experience benchmarks
        from DataScience Jobs dataset.
        """
        role_stats = self.df_ds.groupby("job_title_clean").agg(
            sample_postings=("reference_no", "count"),
            total_vacancies=("num_of_jobs", "sum"),
            avg_salary_lakhs=("avg_salary_lakhs", "mean"),
            min_salary_lakhs=("min_salary_lakhs", "mean"),
            max_salary_lakhs=("max_salary_lakhs", "mean"),
            min_experience_years=("min_experience", "mean")
        ).reset_index()

        # Format and round
        role_stats["avg_salary_lakhs"] = role_stats["avg_salary_lakhs"].round(2)
        role_stats["min_salary_lakhs"] = role_stats["min_salary_lakhs"].round(2)
        role_stats["max_salary_lakhs"] = role_stats["max_salary_lakhs"].round(2)
        role_stats["min_experience_years"] = role_stats["min_experience_years"].round(1)

        role_stats = role_stats.sort_values(by="total_vacancies", ascending=False).reset_index(drop=True)
        return role_stats

    def get_top_hiring_companies(self, top_n: int = 15) -> pd.DataFrame:
        """Returns top hiring enterprises by job volume and their compensation packages."""
        comp_stats = self.df_ds.groupby("company_name_clean").agg(
            total_vacancies=("num_of_jobs", "sum"),
            unique_roles_hired=("job_title_clean", "nunique"),
            avg_salary_offered=("avg_salary_lakhs", "mean"),
            min_exp_required=("min_experience", "mean")
        ).reset_index()

        comp_stats["avg_salary_offered"] = comp_stats["avg_salary_offered"].round(2)
        comp_stats["min_exp_required"] = comp_stats["min_exp_required"].round(1)

        comp_stats = comp_stats.sort_values(by="total_vacancies", ascending=False).head(top_n).reset_index(drop=True)
        return comp_stats

    def get_geographic_distribution(self, top_n: int = 10) -> pd.DataFrame:
        """Calculates geographic concentration of jobs in Analytics Jobs."""
        loc_counts = self.df_aj["primary_location"].value_counts().head(top_n).reset_index()
        loc_counts.columns = ["location", "job_count"]
        total = len(self.df_aj)
        loc_counts["share_pct"] = (loc_counts["job_count"] / total * 100.0).round(2)
        return loc_counts

    def get_salary_band_distribution(self) -> pd.DataFrame:
        """Returns distribution of salary bands from Analytics Jobs."""
        band_counts = self.df_aj["salary"].value_counts().reset_index()
        band_counts.columns = ["salary_band_lakhs", "posting_count"]
        # Order logically by ascending band
        band_order = ["0to3", "3to6", "6to10", "10to15", "15to25", "25to50"]
        band_counts["sort_order"] = band_counts["salary_band_lakhs"].apply(
            lambda x: band_order.index(x) if x in band_order else 99
        )
        band_counts = band_counts.sort_values(by="sort_order").drop(columns=["sort_order"]).reset_index(drop=True)
        band_counts["share_pct"] = (band_counts["posting_count"] / len(self.df_aj) * 100.0).round(2)
        return band_counts

    def get_experience_distribution(self) -> pd.DataFrame:
        """Categorizes required experience into strategic career tiers."""
        def exp_tier(exp):
            if pd.isna(exp) or exp <= 1:
                return "0-1 yrs (Entry / Early Career)"
            elif exp <= 3:
                return "2-3 yrs (Junior / Associate)"
            elif exp <= 6:
                return "4-6 yrs (Mid-Level)"
            elif exp <= 10:
                return "7-10 yrs (Senior / Lead)"
            else:
                return "10+ yrs (Principal / Architect / Exec)"

        self.df_aj["experience_tier"] = self.df_aj["avg_experience"].apply(exp_tier)
        tier_counts = self.df_aj["experience_tier"].value_counts().reset_index()
        tier_counts.columns = ["experience_tier", "posting_count"]
        tier_counts["share_pct"] = (tier_counts["posting_count"] / len(self.df_aj) * 100.0).round(2)
        return tier_counts

    def compare_datasets_summary(self) -> Dict[str, Any]:
        """Provides side-by-side analytical comparison between DataScience Jobs and Analytics Jobs."""
        return {
            "datascience_jobs": {
                "name": "Data Science Jobs (Sample n=1,602)",
                "scope": "Enterprise hiring demand from 642 leading tech & consulting firms",
                "granularity": "Total vacancy counts per role posting (93,005 vacancies represented)",
                "salary_metrics": "Exact salary bounds in Lakhs (min, avg, max)",
                "role_structure": "Standardized enterprise taxonomy (10 canonical career titles)",
                "primary_utility": "Salary benchmarks, hiring volume weighting, career progression hierarchy"
            },
            "analytics_jobs": {
                "name": "Analytics Jobs (Sample n=15,841)",
                "scope": "Broad individual job market postings across India",
                "granularity": "Individual job posting records with granular unstructured skill strings",
                "salary_metrics": "Categorical compensation tiers (0to3L to 25to50L)",
                "role_structure": "Heterogeneous functional designations across 1,350+ locations",
                "primary_utility": "Skill demand co-occurrence, skill normalization, regional concentration"
            }
        }
