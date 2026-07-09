"""Componente Tipos — barras horizontales top 12 + pie chart doméstico."""

from __future__ import annotations

import streamlit as st
from plotly.graph_objects import Bar, Figure, Pie

from src.dashboard.components.styles import (
    COLOR_GRAY,
    COLOR_RED,
    COLOR_TEAL,
    PLOTLY_LAYOUT,
)
from src.storage.duckdb_manager import DuckDBManager


@st.fragment
def render_tipos(
    manager: DuckDBManager,
    year_start: int,
    year_end: int,
) -> None:
    """Renderiza top 12 tipos de crimen y split doméstico."""
    st.markdown("<p class='section-title'>Tipos de crimen</p>", unsafe_allow_html=True)

    df_type = manager.get_crimes_by_type(top_n=12)
    df_domestic = manager.get_domestic_split()

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = Figure(
            Bar(
                x=df_type["total_crimes"].to_list(),
                y=df_type["primary_type"].to_list(),
                orientation="h",
                marker_color=COLOR_RED,
                opacity=0.85,
            )
        )

        # 1. Aplicamos primero el layout base global
        fig.update_layout(**PLOTLY_LAYOUT)

        # 2. Modificamos o sobrescribimos los elementos específicos de este gráfico
        fig.update_layout(
            title={
                "text": "Top 12 Tipos de Crimen",
                "font": {"size": 12, "color": COLOR_GRAY},
            },
            yaxis={
                "autorange": "reversed",
                "gridcolor": "#1E2A3B",
                "tickfont": {"color": COLOR_GRAY, "size": 10},
            },
            showlegend=False,
        )

        # Cambiado use_container_width=True por width="stretch" (Estándar 2026)
        st.plotly_chart(fig, width="stretch")

    with col2:
        labels = [
            "Doméstico" if d else "No Doméstico"
            for d in df_domestic["domestic"].to_list()
        ]

        fig2 = Figure(
            Pie(
                labels=labels,
                values=df_domestic["total_crimes"].to_list(),
                hole=0.5,
                marker_colors=[COLOR_RED, COLOR_TEAL],
                textfont={"color": "#F1FAEE", "size": 11},
            )
        )

        # Aplicamos layout base y luego el título específico
        fig2.update_layout(**PLOTLY_LAYOUT)
        fig2.update_layout(
            title={
                "text": "Crímenes Domésticos",
                "font": {"size": 12, "color": COLOR_GRAY},
            },
        )

        # Cambiado use_container_width=True por width="stretch" (Estándar 2026)
        st.plotly_chart(fig2, width="stretch")
