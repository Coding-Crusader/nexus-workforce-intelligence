"""
NEXUS Frontend Styles & Theming
Custom CSS definitions for a clean, accessible, executive-grade interface
designed for non-technical users with progressive disclosure for evaluators.
"""

CUSTOM_CSS = """
<style>
/* Main typography and container spacing */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1e293b;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Header branding banner */
.nexus-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 20px;
    color: #f8fafc;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
}

.nexus-header h1 {
    font-size: 2.3rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin: 0;
    color: #ffffff;
}

.nexus-header .subtitle {
    font-size: 1.1rem;
    font-weight: 400;
    color: #94a3b8;
    margin-top: 6px;
    letter-spacing: -0.01em;
}

.nexus-tagline {
    font-size: 0.88rem;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 6px;
    display: inline-block;
    padding: 4px 12px;
    margin-top: 12px;
    font-weight: 500;
}

/* Journey Stepper Bar */
.stepper-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    overflow-x: auto;
}

.step-item {
    display: flex;
    align-items: center;
    font-size: 0.85rem;
    font-weight: 500;
    color: #64748b;
    white-space: nowrap;
}

.step-item.active {
    color: #0284c7;
    font-weight: 700;
}

.step-item.completed {
    color: #10b981;
    font-weight: 600;
}

.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: #e2e8f0;
    color: #475569;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 8px;
}

.step-item.active .step-number {
    background: #0284c7;
    color: #ffffff;
}

.step-item.completed .step-number {
    background: #10b981;
    color: #ffffff;
}

.step-arrow {
    margin: 0 8px;
    color: #cbd5e1;
    font-size: 0.8rem;
}

/* Executive KPI metric cards */
.kpi-container {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.kpi-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}

.kpi-subtext {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 4px;
}

/* Micro-explanation Box */
.micro-explain {
    background: #f8fafc;
    border-left: 3px solid #0284c7;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 0.86rem;
    color: #334155;
    margin-top: 8px;
    margin-bottom: 16px;
    line-height: 1.45;
}

.micro-explain strong {
    color: #0f172a;
}

/* Opportunity Cards */
.opp-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.opp-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
}

.opp-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.opp-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
}

.opp-meta {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 12px;
}

.opp-fit-bar-bg {
    width: 100%;
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 14px;
}

.opp-fit-bar-fill {
    height: 100%;
    background: #0284c7;
    border-radius: 4px;
}

.opp-skills-row {
    display: flex;
    gap: 16px;
    font-size: 0.85rem;
    line-height: 1.4;
}

.opp-skills-col {
    flex: 1;
}

.opp-skills-label {
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}

.opp-skills-label.strengths {
    color: #15803d;
}

.opp-skills-label.gaps {
    color: #b91c1c;
}

/* Opportunity Status Badges */
.badge-reachable {
    background-color: #dcfce7;
    color: #15803d;
    font-weight: 600;
    font-size: 0.75rem;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid #bbf7d0;
    display: inline-block;
}

.badge-stretch {
    background-color: #fef9c3;
    color: #a16207;
    font-weight: 600;
    font-size: 0.75rem;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid #fef08a;
    display: inline-block;
}

.badge-blocked {
    background-color: #fee2e2;
    color: #b91c1c;
    font-weight: 600;
    font-size: 0.75rem;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid #fecaca;
    display: inline-block;
}

/* Hero Bottleneck Box */
.hero-gap-box {
    background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
    border: 1px solid #fecdd3;
    border-left: 6px solid #e11d48;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 24px;
}

.hero-gap-tag {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #e11d48;
    margin-bottom: 4px;
}

.hero-gap-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #9f1239;
    margin-bottom: 8px;
}

.hero-gap-desc {
    font-size: 0.95rem;
    color: #4c0519;
    line-height: 1.5;
    margin-bottom: 14px;
}

.hero-gap-bullet {
    display: flex;
    align-items: flex-start;
    font-size: 0.9rem;
    color: #881337;
    margin-bottom: 6px;
}

/* Pathway Decision Cards */
.pathway-card {
    background: #ffffff;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 22px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    transition: all 0.2s ease;
}

.pathway-card:hover {
    border-color: #0284c7;
    box-shadow: 0 6px 16px rgba(2, 132, 199, 0.1);
}

.pathway-card.selected {
    border-color: #0284c7;
    background: #f0f9ff;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.15);
}

.pathway-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: #e0f2fe;
    color: #0369a1;
    margin-bottom: 8px;
}

.pathway-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 8px;
}

.pathway-desc {
    font-size: 0.88rem;
    color: #475569;
    line-height: 1.45;
    margin-bottom: 12px;
}

.pathway-impact {
    font-size: 0.82rem;
    color: #0369a1;
    background: rgba(2, 132, 199, 0.08);
    padding: 8px 12px;
    border-radius: 6px;
    font-weight: 500;
    margin-top: auto;
}

/* Winner Futures Box */
.futures-winner-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
    border: 2px solid #86efac;
    border-radius: 12px;
    padding: 26px;
    margin-top: 20px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);
}

.futures-winner-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #15803d;
    margin-bottom: 6px;
}

.futures-winner-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #14532d;
    margin-bottom: 12px;
}

.futures-winner-metrics {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    margin-top: 14px;
}

.futures-stat {
    background: #ffffff;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 10px 16px;
    flex: 1;
    min-width: 140px;
}

.futures-stat-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #15803d;
}

.futures-stat-label {
    font-size: 0.75rem;
    color: #4b5563;
    font-weight: 500;
}

/* Judge / Analytical Disclosure Accordion */
.judge-box {
    background: #f8fafc;
    border: 1px dashed #94a3b8;
    border-radius: 8px;
    padding: 16px;
    margin-top: 14px;
    margin-bottom: 14px;
}

.judge-tag {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #475569;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}

/* Evidence and Provenance styling */
.evidence-pill {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-right: 6px;
}
.evidence-pill.resume { background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }
.evidence-pill.github { background: #f1f5f9; color: #0f172a; border: 1px solid #cbd5e1; }
.evidence-pill.skill_platform { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
.evidence-pill.portfolio { background: #ede9fe; color: #6d28d9; border: 1px solid #ddd6fe; }
.evidence-pill.self_declared { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }

.provenance-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.provenance-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.provenance-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
}

.why-box {
    background: #f8fafc;
    border-left: 3px solid #10b981;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 0.84rem;
    color: #334155;
    margin-top: 8px;
    line-height: 1.45;
}

.why-box ul {
    margin: 4px 0 0 16px;
    padding: 0;
}

.why-box li {
    margin-bottom: 3px;
}

/* Custom button touch */
.stButton>button {
    font-weight: 600;
    border-radius: 6px;
    padding: 8px 18px;
    transition: all 0.15s ease-in-out;
}
</style>
"""

def render_kpi(title: str, value: str, subtext: str = "") -> str:
    """Returns HTML for an executive metric card."""
    sub_html = f'<div class="kpi-subtext">{subtext}</div>' if subtext else ""
    return f"""
    <div class="kpi-container">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """

def render_stepper(current_step: int) -> str:
    """Renders a friendly progress indicator bar."""
    steps = [
        (1, "1. Identify"),
        (2, "2. Add Evidence"),
        (3, "3. Your Profile"),
        (4, "4. Market Fit"),
        (5, "5. Model Signals"),
        (6, "6. Highest Gap"),
        (7, "7. What-If Futures")
    ]
    
    html = ['<div class="stepper-container">']
    for idx, (num, name) in enumerate(steps):
        state = ""
        if num == current_step:
            state = "active"
        elif num < current_step:
            state = "completed"
            
        icon = "✓" if num < current_step else str(num)
        html.append(f"""
        <div class="step-item {state}">
            <span class="step-number">{icon}</span>
            <span>{name}</span>
        </div>
        """)
        if idx < len(steps) - 1:
            html.append('<span class="step-arrow">→</span>')
            
    html.append('</div>')
    return "".join(html)

def render_micro_explanation(text: str, strong_prefix: str = "What this means:") -> str:
    """Renders a clean micro-explanation box."""
    return f"""
    <div class="micro-explain">
        <strong>{strong_prefix}</strong> {text}
    </div>
    """
