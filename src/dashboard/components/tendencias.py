"""Componente Tendencias — línea doble eje por año + barras por hora."""

from __future__ import annotations

import streamlit as st
from plotly.graph_objects import Bar, Figure, Scatter
from plotly.subplots import make_subplots

from src.dashboard.components.styles import (
    COLOR_GRAY,
    COLOR_RED,
    COLOR_TEAL,
    PLOTLY_LAYOUT,
)
from src.storage.duckdb_manager import DuckDBManager


@st.fragment
def render_tendencias(
    manager: DuckDBManager,
    year_start: int,
    year_end: int,
    community_area: str | None,
) -> None:
    """Renderiza gráfico de tendencias anuales y distribución por hora."""
    st.markdown(
        "<p class='section-title'>Tendencias temporales</p>", unsafe_allow_html=True
    )

    df_year = manager.get_crimes_by_year(year_start, year_end, community_area)
    df_hour = manager.get_crimes_by_hour(year_start, year_end)

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            Bar(
                x=df_year["year"].to_list(),
                y=df_year["total_crimes"].to_list(),
                name="Total Crímenes",
                marker_color=COLOR_RED,
                opacity=0.8,
            ),
            secondary_y=False,
        )

        fig.add_trace(
            Scatter(
                x=df_year["year"].to_list(),
                y=df_year["arrest_rate"].to_list(),
                name="Tasa Arresto %",
                line={"color": COLOR_TEAL, "width": 2},
                mode="lines+markers",
                marker={"size": 4},
            ),
            secondary_y=True,
        )

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title={
                "text": "Crímenes y Tasa de Arresto por Año",
                "font": {"size": 12, "color": COLOR_GRAY},
            },
            hovermode="x unified",
        )
        fig.update_yaxes(
            title_text="Total Crímenes",
            gridcolor="#1E2A3B",
            tickfont={"color": COLOR_GRAY, "size": 10},
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Tasa Arresto %",
            gridcolor="#1E2A3B",
            tickfont={"color": COLOR_GRAY, "size": 10},
            secondary_y=True,
        )

        st.plotly_chart(fig, width="stretch")

    with col2:
        night_hours = list(range(0, 6)) + list(range(22, 24))
        fig2 = Figure(
            Bar(
                x=df_hour["hour"].to_list(),
                y=df_hour["total_crimes"].to_list(),
                marker_color=[
                    COLOR_RED if h in night_hours else COLOR_TEAL
                    for h in df_hour["hour"].to_list()
                ],
                opacity=0.85,
            )
        )

        fig2.update_layout(
            **PLOTLY_LAYOUT,
            title={
                "text": "Crímenes por Hora del Día",
                "font": {"size": 12, "color": COLOR_GRAY},
            },
            xaxis_title="Hora",
            showlegend=False,
        )

        st.plotly_chart(fig2, width="stretch")
