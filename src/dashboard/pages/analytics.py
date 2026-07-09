"""Pagina 1 — Dashboard analítico de crímenes."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dashboard.community_areas import NAME_TO_AREA, code_to_name
from src.dashboard.components.kpis import render_kpis
from src.dashboard.components.mapa import render_mapa
from src.dashboard.components.styles import (
    COLOR_GRAY,
    COLOR_RED,
    COLOR_TEAL,
    PLOTLY_LAYOUT,
)
from src.dashboard.components.tipos import render_tipos
from src.storage.duckdb_manager import DuckDBManager


@st.cache_data(ttl=3600)
def _get_community_area_names(_manager: DuckDBManager) -> list[str]:
    """Retorna nombres de community areas presentes en el dataset, cacheados 1h."""
    df = _manager.get_heatmap_data(year_start=2001, year_end=2026)
    names = {code_to_name(c) for c in df["community_area"].to_list()}
    return sorted(names)


@st.cache_data(ttl=3600)
def _get_crime_types(_manager: DuckDBManager) -> list[str]:
    """Retorna los tipos de crimen presentes en el dataset, cacheados 1h."""
    df = _manager.get_crimes_by_type(top_n=50)
    return df["primary_type"].to_list()


def _render_filter_bar(
    community_areas: list[str],
    crime_types: list[str],
) -> tuple[int, int, str | None, str | None, str | None]:
    """Barra de filtros horizontal estilo Power BI. Retorna (year_start, year_end, area, crime_type, domestic)."""
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([2.4, 1.5, 1.5, 1.2, 0.7])

        with c1:
            st.markdown(
                '<div class="filter-label">Rango de años</div>', unsafe_allow_html=True
            )
            year_range: tuple[int, int] = st.slider(
                "Años",
                2001,
                2026,
                value=st.session_state.get("f_years", (2001, 2026)),
                label_visibility="collapsed",
                key="f_years",
            )

        with c2:
            st.markdown(
                '<div class="filter-label">Community Area</div>', unsafe_allow_html=True
            )
            area_name: str = st.selectbox(
                "Área",
                ["Todas"] + community_areas,
                index=0,
                label_visibility="collapsed",
                key="f_area",
            )

        with c3:
            st.markdown(
                '<div class="filter-label">Tipo de crimen</div>', unsafe_allow_html=True
            )
            crime_type: str = st.selectbox(
                "Tipo",
                ["Todos"] + crime_types,
                index=0,
                label_visibility="collapsed",
                key="f_type",
            )

        with c4:
            st.markdown(
                '<div class="filter-label">Doméstico</div>', unsafe_allow_html=True
            )
            domestic: str = st.selectbox(
                "Doméstico",
                ["Todos", "Sí", "No"],
                index=0,
                label_visibility="collapsed",
                key="f_dom",
            )

        with c5:
            st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
            if st.button("🔄 Limpiar", width="stretch", key="f_clear"):
                for k in ("f_years", "f_area", "f_type", "f_dom"):
                    st.session_state.pop(k, None)
                st.rerun()

    year_start, year_end = year_range
    area_filter: str | None = (
        None if area_name == "Todas" else str(NAME_TO_AREA.get(area_name, area_name))
    )
    type_filter: str | None = None if crime_type == "Todos" else crime_type
    dom_filter: str | None = None if domestic == "Todos" else domestic

    return year_start, year_end, area_filter, type_filter, dom_filter


def _render_trends(
    manager: DuckDBManager,
    year_start: int,
    year_end: int,
    community_area: str | None,
    primary_type: str | None = None,
    domestic: bool | None = None,
) -> None:
    """Gráfico dual eje: barras crímenes + línea tasa arresto."""
    df = manager.get_crimes_by_year(
        year_start, year_end, community_area, primary_type, domestic
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=df["year"].to_list(),
            y=df["total_crimes"].to_list(),
            name="Total Crímenes",
            marker_color=COLOR_RED,
            opacity=0.88,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["year"].to_list(),
            y=df["arrest_rate"].to_list(),
            name="Tasa Arresto %",
            line={"color": COLOR_TEAL, "width": 2.5},
            mode="lines+markers",
            marker={"size": 4},
        ),
        secondary_y=True,
    )
    fig.update_layout(**PLOTLY_LAYOUT, hovermode="x unified")
    fig.update_layout(
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": COLOR_GRAY, "size": 10},
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    fig.update_yaxes(
        title_text="Total Crímenes",
        gridcolor="#1E2A3B",
        tickfont={"color": COLOR_GRAY, "size": 10},
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Tasa Arresto %",
        gridcolor=None,
        showgrid=False,
        tickfont={"color": COLOR_GRAY, "size": 10},
        secondary_y=True,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render(manager: DuckDBManager) -> None:
    """Renderiza la página completa de analytics."""
    # ── Encabezado ───────────────────────────────────────────────────────
    st.markdown("<p class='page-title'>Analytics dashboard</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='page-subtitle'>2001 – 2026 · 8,464,732 registros · Vista general</p>",
        unsafe_allow_html=True,
    )

    # ── Filtros ──────────────────────────────────────────────────────────
    community_areas = _get_community_area_names(manager)
    crime_types = _get_crime_types(manager)
    year_start, year_end, area_filter, type_filter, dom_filter = _render_filter_bar(
        community_areas, crime_types
    )

    # Convierte "Sí"/"No" → bool para el manager
    dom_bool: bool | None = None if dom_filter is None else (dom_filter == "Sí")

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────────────────
    render_kpis(manager, year_start, year_end, area_filter, type_filter, dom_bool)

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # ── Mapa + Tendencias ────────────────────────────────────────────────
    col_map, col_trend = st.columns(2)

    with col_map, st.container(border=True):
        render_mapa(manager, year_start, year_end)

    with col_trend, st.container(border=True):
        st.markdown(
            "<p class='section-title'>Crímenes y Tasa de Arresto por Año</p>",
            unsafe_allow_html=True,
        )
        _render_trends(
            manager, year_start, year_end, area_filter, type_filter, dom_bool
        )

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # ── Tipos ────────────────────────────────────────────────────────────
    render_tipos(manager, year_start, year_end)
