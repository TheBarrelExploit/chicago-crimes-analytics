"""CSS global y constantes de diseño Plotly — paleta "Chicago Night"."""

from typing import Any

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0E1A;
    color: #F1FAEE;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1E2A3B;
}
.dash-header {
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid #1E2A3B;
    margin-bottom: 1.5rem;
}
.dash-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #F1FAEE;
    letter-spacing: -0.02em;
    margin: 0;
}
.dash-subtitle {
    font-size: 0.8rem;
    color: #8D99AE;
    margin-top: 0.2rem;
    font-family: 'IBM Plex Mono', monospace;
}
.kpi-card {
    background: #111827;
    border: 1px solid #1E2A3B;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
}
.kpi-label {
    font-size: 0.7rem;
    color: #8D99AE;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #F1FAEE;
    line-height: 1;
}
.kpi-value.accent-red  { color: #E63946; }
.kpi-value.accent-teal { color: #2EC4B6; }
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #8D99AE;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1E2A3B;
}
.sidebar-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #8D99AE;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #F1FAEE;
    margin: 0 0 0.2rem 0;
    letter-spacing: -0.02em;
}
.page-subtitle {
    font-size: 0.82rem;
    color: #8D99AE;
    margin: 0 0 1rem 0;
    font-family: 'IBM Plex Mono', monospace;
}
.filter-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #5b6478;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 2px;
}
.placeholder-box {
    background: #111827;
    border: 1px dashed #1E2A3B;
    border-radius: 8px;
    padding: 3rem 2rem;
    text-align: center;
    color: #8D99AE;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.8;
}
#MainMenu, footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stSidebarNav"] { display: none !important; }
/* Ocultar el header superior de Streamlit por completo */
header[data-testid="stHeader"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
}
/* Sin padding top en el contenido principal */
[data-testid="stAppViewContainer"] > section:last-child {
    padding-top: 1rem !important;
}
/* Header del sidebar — fondo oscuro */
[data-testid="stSidebarHeader"] {
    background-color: #111827 !important;
    border-bottom: 1px solid #1E2A3B !important;
}
/* Botón colapsar nativo — estilo consistente */
[data-testid="stSidebarCollapseButton"] button {
    color: #8D99AE !important;
    background-color: transparent !important;
}
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapseButton"] svg path {
    fill: #8D99AE !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:not(:first-child) {
    border-left: 0.5px solid #1e2738;
    padding-left: 16px;
}
</style>
"""

COLOR_RED = "#E63946"
COLOR_TEAL = "#2EC4B6"
COLOR_GRAY = "#8D99AE"

PLOTLY_LAYOUT: dict[str, Any] = {
    "paper_bgcolor": "#111827",
    "plot_bgcolor": "#111827",
    "font": {"family": "Inter, sans-serif", "color": "#F1FAEE", "size": 12},
    "margin": {"t": 30, "b": 30, "l": 10, "r": 10},
    "xaxis": {
        "gridcolor": "#1E2A3B",
        "tickcolor": "#8D99AE",
        "tickfont": {"color": "#8D99AE", "size": 10},
        "showline": False,
    },
    "yaxis": {
        "gridcolor": "#1E2A3B",
        "tickcolor": "#8D99AE",
        "tickfont": {"color": "#8D99AE", "size": 10},
        "showline": False,
    },
    "legend": {
        "bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#8D99AE", "size": 10},
    },
}
