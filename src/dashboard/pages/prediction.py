"""Página de Predicción — formulario + gauge de probabilidad de arresto."""

from __future__ import annotations

import datetime
import logging

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.community_areas import COMMUNITY_AREAS, NAME_TO_AREA
from src.dashboard.components.styles import COLOR_TEAL, PLOTLY_LAYOUT
from src.storage.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)

_BADGE_THRESHOLDS = {"Alta": 0.55, "Media": 0.35}
_CHICAGO_LAT = 41.8781
_CHICAGO_LON = -87.6298


def _get_area_centroid(
    manager: DuckDBManager, area_code: int
) -> tuple[float, float]:
    """Returns approximate (lat, lon) centroid for a community area."""
    try:
        import polars as pl

        df = manager.get_heatmap_data(2020, 2025)
        filtered = df.filter(pl.col("community_area").cast(pl.Utf8) == str(area_code))
        if len(filtered) > 0:
            return (float(filtered["lat_centroid"][0]), float(filtered["lon_centroid"][0]))
    except Exception:
        pass
    return (_CHICAGO_LAT, _CHICAGO_LON)


def _arrest_gauge(probability: float) -> go.Figure:
    pct = probability * 100
    if probability >= _BADGE_THRESHOLDS["Alta"]:
        color = "#E63946"
        badge = "Alta"
    elif probability >= _BADGE_THRESHOLDS["Media"]:
        color = "#f5a623"
        badge = "Media"
    else:
        color = COLOR_TEAL
        badge = "Baja"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={
                "suffix": "%",
                "font": {"size": 36, "color": "#F1FAEE",
                         "family": "IBM Plex Mono, monospace"},
            },
            title={
                "text": f"Probabilidad de Arresto — <b>{badge}</b>",
                "font": {"size": 14, "color": "#8D99AE"},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickcolor": "#8D99AE", "tickfont": {"size": 10}},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(46,196,182,0.08)"},
                    {"range": [35, 55], "color": "rgba(245,166,35,0.08)"},
                    {"range": [55, 100], "color": "rgba(230,57,70,0.08)"},
                ],
            },
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=280)
    fig.update_layout(margin={"t": 60, "b": 20, "l": 20, "r": 20})
    return fig


def render(manager: DuckDBManager) -> None:
    """Renders the ML prediction page."""
    st.markdown(
        "<p class='page-title'>Predicción de Arresto</p>"
        "<p class='page-subtitle'>XGBoost · Probabilidad basada en características del crimen</p>",
        unsafe_allow_html=True,
    )

    # Load model (cached) — show error if not yet uploaded to DagsHub
    try:
        from src.ml.predict import load_production_model, predict_arrest_probability

        booster, le_primary, le_location = load_production_model()
        crime_types = sorted(le_primary.classes_.tolist())
        location_types = sorted(le_location.classes_.tolist())
        model_available = True
    except Exception as exc:
        st.error(
            f"Modelo no disponible: {exc}\n\n"
            "Ejecuta `uv run python -m src.ml.upload_initial` para registrar el modelo."
        )
        model_available = False
        crime_types = []
        location_types = []

    area_names = list(COMMUNITY_AREAS.values())

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<p class='section-title'>Parámetros del crimen</p>",
                    unsafe_allow_html=True)
        primary_type = st.selectbox(
            "Tipo de crimen", crime_types, disabled=not model_available, key="pred_type"
        )
        location_description = st.selectbox(
            "Lugar", location_types, disabled=not model_available, key="pred_loc"
        )
        area_name = st.selectbox(
            "Community Area", area_names, disabled=not model_available, key="pred_area"
        )
        district = st.number_input(
            "Distrito", min_value=1, max_value=25, value=1,
            disabled=not model_available, key="pred_district",
        )
        beat = st.number_input(
            "Beat", min_value=100, max_value=2500, value=100, step=1,
            disabled=not model_available, key="pred_beat",
        )
        hour = st.slider(
            "Hora del día", min_value=0, max_value=23, value=12,
            disabled=not model_available, key="pred_hour",
        )
        domestic = st.toggle(
            "¿Crimen doméstico?", value=False,
            disabled=not model_available, key="pred_domestic",
        )
        crime_date = st.date_input(
            "Fecha del crimen",
            value=datetime.date.today(),
            disabled=not model_available,
            key="pred_date",
        )
        predict_btn = st.button(
            "Predecir probabilidad de arresto",
            disabled=not model_available,
            use_container_width=True,
            type="primary",
        )

    with col_result:
        st.markdown("<p class='section-title'>Resultado</p>", unsafe_allow_html=True)

        if predict_btn and model_available:
            area_code = NAME_TO_AREA.get(area_name, 8)
            lat, lon = _get_area_centroid(manager, area_code)

            record = {
                "primary_type": primary_type,
                "location_description": location_description,
                "domestic": domestic,
                "year": crime_date.year,
                "hour": hour,
                "district": int(district),
                "community_area": area_code,
                "latitude": lat,
                "beat": int(beat),
                "longitude": lon,
                "month": crime_date.month,
                "quarter": (crime_date.month - 1) // 3 + 1,
                "weekday": crime_date.weekday(),
            }

            try:
                probability = predict_arrest_probability(record)
                st.plotly_chart(
                    _arrest_gauge(probability),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            except Exception as exc:
                st.error(f"Error al predecir: {exc}")
        else:
            st.markdown(
                "<div class='placeholder-box' style='margin-top:2rem;'>"
                "Configura los parámetros y haz clic en<br>"
                "<b>Predecir probabilidad de arresto</b>"
                "</div>",
                unsafe_allow_html=True,
            )
