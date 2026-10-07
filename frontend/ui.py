"""
NEXUS Frontend UI Components & Visualizations
High-density, executive-grade Plotly visualizations and layout renderers.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.styles import render_kpi

# Refined corporate slate palette
COLORS = {
    "primary": "#0284c7",
    "primary_dark": "#0369a1",
    "secondary": "#475569",
    "accent": "#0ea5e9",
    "success": "#16a34a",
    "warning": "#d97706",
    "danger": "#dc2626",
    "slate_bg": "#f8fafc",
    "grid": "#e2e8f0"
}

def render_nexus_header():
    """Renders the executive branding header."""
    st.markdown("""
    <div class="nexus-header">
        <h1>NEXUS</h1>
        <div class="subtitle">Workforce Transition Intelligence & Counterfactual Opportunity Modeling</div>
        <div class="nexus-tagline">"Don't just measure where talent is. Model where talent can go."</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# Market Visualizations
# ==========================================

def plot_skill_demand_bar(df_skills: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Plots canonical skill demand frequency and job coverage percentage."""
    df_plot = df_skills.head(top_n).sort_values(by="demand_count", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df_plot["demand_count"],
        y=df_plot["display_name"],
        orientation="h",
        marker=dict(color=COLORS["primary"], line=dict(color=COLORS["primary_dark"], width=1)),
        text=[f"{count:,} ({pct}%)" for count, pct in zip(df_plot["demand_count"], df_plot["job_coverage_pct"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Job Postings: %{x:,}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Market Demand Frequency by Canonical Skill Family</b>",
        xaxis_title="Number of Analyzed Job Postings",
        yaxis_title="",
        margin=dict(l=10, r=40, t=40, b=30),
        height=400,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"])
    )
    return fig

def plot_role_benchmarks_bubble(df_benchmarks: pd.DataFrame) -> go.Figure:
    """Plots role market landscape: Experience vs Average Salary with Vacancy bubble size."""
    fig = px.scatter(
        df_benchmarks,
        x="min_experience_years",
        y="avg_salary_lakhs",
        size="total_vacancies",
        color="avg_salary_lakhs",
        color_continuous_scale="Blues",
        text="job_title_clean",
        hover_data={
            "job_title_clean": True,
            "min_experience_years": ":.1f yrs",
            "avg_salary_lakhs": ":.2fL",
            "total_vacancies": ":,",
            "sample_postings": True
        },
        labels={
            "min_experience_years": "Min Experience Required (Years)",
            "avg_salary_lakhs": "Average Salary (Lakhs INR)",
            "total_vacancies": "Total Market Vacancies"
        }
    )

    fig.update_traces(
        textposition="top center",
        marker=dict(line=dict(width=1.5, color="#1e293b"))
    )

    fig.update_layout(
        title="<b>Role Landscape: Experience vs Compensation vs Market Demand</b>",
        height=480,
        margin=dict(l=20, r=20, t=50, b=40),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"]),
        yaxis=dict(showgrid=True, gridcolor=COLORS["grid"]),
        coloraxis_showscale=False
    )
    return fig

def plot_location_distribution(df_locs: pd.DataFrame) -> go.Figure:
    """Plots metropolitan hiring concentration."""
    df_plot = df_locs.sort_values(by="job_count", ascending=True)
    fig = go.Figure(go.Bar(
        x=df_plot["job_count"],
        y=df_plot["location"],
        orientation="h",
        marker=dict(color="#334155"),
        text=[f"{c:,} ({p}%)" for c, p in zip(df_plot["job_count"], df_plot["share_pct"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Postings: %{x:,}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Geographic Concentration (Top Metropolitan Hubs)</b>",
        xaxis_title="Job Postings",
        yaxis_title="",
        margin=dict(l=10, r=50, t=40, b=30),
        height=380,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"])
    )
    return fig

def plot_salary_bands_donut(df_bands: pd.DataFrame) -> go.Figure:
    """Plots salary band breakdown from Analytics Jobs."""
    fig = go.Figure(go.Pie(
        labels=[f"{b} Lakhs" for b in df_bands["salary_band_lakhs"]],
        values=df_bands["posting_count"],
        hole=0.55,
        marker=dict(colors=["#0284c7", "#0ea5e9", "#38bdf8", "#64748b", "#475569", "#1e293b"]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Postings: %{value:,} (%{percent})<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Compensation Bands (Analytics Jobs)</b>",
        height=380,
        margin=dict(l=10, r=10, t=40, b=20),
        showlegend=False
    )
    return fig

# ==========================================
# ML & Success Visualizations
# ==========================================

def plot_coefficients_bar(df_coefs: pd.DataFrame, title: str, x_col: str = "coefficient", y_col: str = "display_name") -> go.Figure:
    """Plots horizontal bar chart of logistic regression coefficients."""
    df_plot = df_coefs.sort_values(by=x_col, ascending=True)
    colors = [COLORS["success"] if val >= 0 else COLORS["danger"] for val in df_plot[x_col]]

    fig = go.Figure(go.Bar(
        x=df_plot[x_col],
        y=df_plot[y_col],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.3f}" for v in df_plot[x_col]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Logistic Regression Weight (Log-Odds Impact)",
        yaxis_title="",
        margin=dict(l=10, r=40, t=40, b=30),
        height=340,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=True, zerolinecolor="#64748b")
    )
    return fig

def plot_confusion_matrix_heatmap(cm: List[List[int]], labels: List[str], title: str) -> go.Figure:
    """Plots confusion matrix heatmap."""
    z = np.array(cm)
    fig = px.imshow(
        z,
        text_auto=True,
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        title=f"<b>{title}</b>",
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_showscale=False
    )
    return fig

# ==========================================
# Candidate & Opportunity Visualizations
# ==========================================

def plot_capability_radar(capabilities: Dict[str, float], title: str = "Candidate Capability Profile") -> go.Figure:
    """Plots radar/spider chart of candidate capability scores."""
    categories = [k.replace("_", " ").title() for k in capabilities.keys()]
    values = list(capabilities.values())

    # Close the radar loop
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(2, 132, 199, 0.2)",
        line=dict(color=COLORS["primary"], width=2),
        name="Capability Level"
    ))

    fig.update_layout(
        title=f"<b>{title}</b>",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.0, 1.0],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                gridcolor=COLORS["grid"]
            )
        ),
        height=380,
        margin=dict(l=40, r=40, t=50, b=30),
        showlegend=False
    )
    return fig

def plot_simulation_waterfall(comparison_df: pd.DataFrame) -> go.Figure:
    """Plots compatibility score change per role before vs after intervention."""
    df_sorted = comparison_df.sort_values(by="delta_score_pct", ascending=True)

    fig = go.Figure()

    # Score before
    fig.add_trace(go.Bar(
        y=df_sorted["role_name"],
        x=df_sorted["score_before"],
        name="Baseline Fit (%)",
        orientation="h",
        marker=dict(color="#94a3b8")
    ))

    # Delta score
    fig.add_trace(go.Bar(
        y=df_sorted["role_name"],
        x=df_sorted["delta_score_pct"],
        name="Intervention Gain (+%)",
        orientation="h",
        marker=dict(color=COLORS["primary"])
    ))

    fig.update_layout(
        title="<b>Counterfactual Role Fit: Baseline vs Post-Intervention Score Gain</b>",
        barmode="stack",
        xaxis_title="Compatibility Score (%)",
        yaxis_title="",
        height=480,
        margin=dict(l=20, r=20, t=50, b=30),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_bottlenecks_bar(df_bn: pd.DataFrame) -> go.Figure:
    """Plots bottleneck capabilities ranked by transition leverage."""
    df_plot = df_bn.head(7).sort_values(by="priority_score", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_plot["priority_score"],
        y=df_plot["display_name"],
        orientation="h",
        marker=dict(color="#e11d48"),
        text=[f"Score: {s:.3f} | Gap: {g:.2f}" for s, g in zip(df_plot["priority_score"], df_plot["average_gap_in_target_roles"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Bottleneck Priority: %{x:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Top Capability Bottlenecks (Gap × Market Demand × Opportunity Impact)</b>",
        xaxis_title="Transition Leverage Score (Higher = Greater Bottleneck)",
        yaxis_title="",
        margin=dict(l=10, r=80, t=40, b=30),
        height=340,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"])
    )
    return fig

def plot_three_futures_comparison(sim_results: List[Dict[str, Any]]) -> go.Figure:
    """Plots side-by-side comparison of the 3 Simulated Futures (Presentation Slide 09)."""
    names = [s["intervention"].get("short_name", s["intervention"]["name"]) for s in sim_results]
    expansions = [s["opportunity_expansion_pct"] for s in sim_results]
    salaries = [s["salary_growth_lakhs"] for s in sim_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=expansions,
        name="Opportunity Expansion (%)",
        marker=dict(color=COLORS["primary"]),
        text=[f"+{e:.1f}%" for e in expansions],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        x=names,
        y=salaries,
        name="Salary Lift (+Lakhs)",
        marker=dict(color=COLORS["success"]),
        text=[f"+{sl:.2f}L" for sl in salaries],
        textposition="outside"
    ))

    fig.update_layout(
        title="<b>The 3 Simulated Futures: Opportunity Expansion & Compensation Lift (Slide 09)</b>",
        barmode="group",
        height=380,
        margin=dict(l=20, r=20, t=50, b=30),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=COLORS["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

