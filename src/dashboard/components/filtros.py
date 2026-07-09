"""Barra de filtros horizontal, estilo Power BI."""

from __future__ import annotations

import streamlit as st


def render_filtros_horizontal(
    year_min: int = 2001,
    year_max: int = 2026,
    community_areas: list[str] | None = None,
    crime_types: list[str] | None = None,
) -> tuple[int, int, str, str, str]:
    """Renderiza la barra de filtros y devuelve los valores seleccionados."""
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([2.2, 1.4, 1.4, 1.2, 0.7])

        with c1:
            year_start, year_end = st.slider(
                "Años",
                year_min,
                year_max,
                (year_min, year_max),
                key="f_years",
            )
        with c2:
            area = st.selectbox(
                "Community area", ["Todas"] + (community_areas or []), key="f_area"
            )
        with c3:
            crime_type = st.selectbox(
                "Tipo de crimen", ["Todos"] + (crime_types or []), key="f_type"
            )
        with c4:
            domestic = st.selectbox("Doméstico", ["Todos", "Sí", "No"], key="f_dom")
        with c5:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🔄 Limpiar", width="stretch", key="f_clear"):
                for k in ("f_years", "f_area", "f_type", "f_dom"):
                    st.session_state.pop(k, None)
                st.rerun()

    return year_start, year_end, area, crime_type, domestic
