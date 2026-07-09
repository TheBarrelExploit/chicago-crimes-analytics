"""Componente KPIs — 4 cards con métricas principales y sparklines."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.components.styles import COLOR_RED, COLOR_TEAL
from src.storage.duckdb_manager import DuckDBManager

_COLOR_ORANGE = "#f5a623"


def fmt_number(n: int | float) -> str:
    """Formatea número con separadores de miles."""
    return f"{int(n):,}"


def fmt_pct(p: float) -> str:
    """Formatea porcentaje con un decimal."""
    return f"{p:.1f}%"


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _sparkline(values: list[float], color: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            y=values,
            mode="lines",
            line={"color": color, "width": 2},
            fill="tozeroy",
            fillcolor=_hex_to_rgba(color, 0.12),
        )
    )
    fig.update_layout(
        height=44,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
        showlegend=False,
    )
    return fig


def _gauge(value: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "suffix": "%",
                "font": {
                    "size": 24,
                    "color": "#F1FAEE",
                    "family": "IBM Plex Mono, monospace",
                },
            },
            gauge={
                "axis": {"range": [0, 40], "visible": False},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
            },
        )
    )
    fig.update_layout(
        height=100,
        margin={"l": 10, "r": 10, "t": 6, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.fragment
def render_kpis(
    manager: DuckDBManager,
    year_start: int,
    year_end: int,
    community_area: str | None,
    primary_type: str | None = None,
    domestic: bool | None = None,
) -> None:
    """Renderiza los 4 KPIs principales con sparklines y gauge."""
    kpis = manager.get_kpis(year_start, year_end, community_area, primary_type, domestic)
    df_year = manager.get_crimes_by_year(year_start, year_end, community_area, primary_type, domestic)

    crimes_series = df_year["total_crimes"].to_list()
    arrests_series = df_year["total_arrests"].to_list()
    domestic_series = df_year["total_domestic"].to_list()

    col1, col2, col3, col4 = st.columns(4)

    with col1, st.container(border=True):
        st.markdown(
            f"""<div class='kpi-label'>Total Crímenes</div>
            <div class='kpi-value accent-red'>{fmt_number(kpis["total_crimes"])}</div>""",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _sparkline(crimes_series, COLOR_RED),
            width="stretch",
            config={"displayModeBar": False},
        )

    with col2, st.container(border=True):
        st.markdown(
            f"""<div class='kpi-label'>Total Arrestos</div>
            <div class='kpi-value accent-teal'>{fmt_number(kpis["total_arrests"])}</div>""",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _sparkline(arrests_series, COLOR_TEAL),
            width="stretch",
            config={"displayModeBar": False},
        )

    with col3, st.container(border=True):
        st.markdown(
            f"""<div class='kpi-label'>Total Doméstico</div>
            <div class='kpi-value' style='color:{_COLOR_ORANGE}'>{fmt_number(kpis["total_domestic"])}</div>""",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _sparkline(domestic_series, _COLOR_ORANGE),
            width="stretch",
            config={"displayModeBar": False},
        )

    with col4, st.container(border=True):
        st.markdown(
            "<div class='kpi-label'>Tasa de Arresto</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _gauge(kpis["arrest_rate"], COLOR_TEAL),
            width="stretch",
            config={"displayModeBar": False},
        )
