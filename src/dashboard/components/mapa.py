"""Componente Mapa — burbujas por community area con escala de color."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.components.styles import COLOR_GRAY, COLOR_RED, COLOR_TEAL
from src.storage.duckdb_manager import DuckDBManager


@st.fragment
def render_mapa(
    manager: DuckDBManager,
    year_start: int,
    year_end: int,
) -> None:
    """Renderiza mapa de burbujas por community area."""
    st.markdown(
        "<p class='section-title'>Distribución geográfica</p>", unsafe_allow_html=True
    )

    df_map = manager.get_heatmap_data(year_start, year_end)
    max_crimes = df_map["total_crimes"].max() or 1

    fig = go.Figure(
        go.Scattermapbox(
            lat=df_map["lat_centroid"].to_list(),
            lon=df_map["lon_centroid"].to_list(),
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=[
                    max(6, min(40, v / max_crimes * 40))
                    for v in df_map["total_crimes"].to_list()
                ],
                color=df_map["total_crimes"].to_list(),
                colorscale=[[0, COLOR_TEAL], [1, COLOR_RED]],
                showscale=True,
                colorbar={
                    "title": {
                        "text": "Crímenes",
                        "font": {"color": COLOR_GRAY, "size": 10},
                    },
                    "tickfont": {"color": COLOR_GRAY, "size": 9},
                    "bgcolor": "#111827",
                },
                opacity=0.85,
            ),
            text=df_map["community_area"].to_list(),
            customdata=df_map["total_crimes"].to_list(),
            hovertemplate="<b>%{text}</b><br>Crímenes: %{customdata:,}<extra></extra>",
        )
    )

    fig.update_layout(
        mapbox={
            "style": "carto-darkmatter",
            "center": {"lat": 41.85, "lon": -87.65},
            "zoom": 9.5,
        },
        paper_bgcolor="#111827",
        margin={"t": 0, "b": 0, "l": 0, "r": 0},
        height=450,
    )

    st.plotly_chart(fig, width="stretch")
