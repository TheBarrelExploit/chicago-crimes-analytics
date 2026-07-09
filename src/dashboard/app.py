"""Chicago Crimes Analytics — Entry Point."""

from __future__ import annotations

import sys
from pathlib import Path

# Agrega la raíz del proyecto a sys.path antes de cualquier import de src.*
# Streamlit solo agrega el directorio del script (src/dashboard/) pero no la raíz.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from streamlit_option_menu import option_menu

from src.dashboard.components.styles import CSS
from src.dashboard.pages import analytics, prediction
from src.storage.duckdb_manager import DuckDBManager

st.set_page_config(
    page_title="Chicago Crimes Analytics",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

_FONT_MONO = "'IBM Plex Mono', monospace"
_FONT_SANS = "'Inter', sans-serif"
_COLOR_DIM = "#5b6478"
_COLOR_TEXT = "#F1FAEE"
_COLOR_MUTED = "#8D99AE"
_COLOR_CRIME = "#E63946"


@st.cache_resource
def _get_manager() -> DuckDBManager:
    """Crea y cachea una instancia de DuckDBManager para toda la sesión."""
    return DuckDBManager()


def _inject_sidebar_toggle() -> None:
    """Inject a floating expand button that appears when the sidebar is collapsed."""
    st.html(
        """
        <script>
        (function() {
            function setup() {
                var sidebar = document.querySelector('[data-testid="stSidebar"]');
                if (!sidebar) return;

                var btn = document.getElementById('__cc_expand__');
                if (!btn) {
                    btn = document.createElement('button');
                    btn.id = '__cc_expand__';
                    btn.title = 'Expandir menú';
                    btn.innerHTML = '&#8250;';
                    btn.style.cssText = [
                        'position:fixed', 'left:0', 'top:50%',
                        'transform:translateY(-50%)',
                        'z-index:99999',
                        'width:20px', 'height:44px',
                        'background:#111827',
                        'color:#8D99AE',
                        'border:1px solid #1E2A3B',
                        'border-left:none',
                        'border-radius:0 6px 6px 0',
                        'cursor:pointer',
                        'font-size:20px',
                        'line-height:44px',
                        'text-align:center',
                        'padding:0',
                        'display:none',
                        'transition:color .15s',
                    ].join(';');
                    btn.onmouseenter = function() { btn.style.color = '#F1FAEE'; };
                    btn.onmouseleave = function() { btn.style.color = '#8D99AE'; };
                    btn.onclick = function() {
                        var real = document.querySelector(
                            '[data-testid="stSidebarCollapseButton"] button'
                        );
                        if (real) real.click();
                    };
                    document.body.appendChild(btn);
                }

                var expanded = sidebar.getAttribute('aria-expanded') === 'true';
                btn.style.display = expanded ? 'none' : 'block';
            }

            setInterval(setup, 400);
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


@st.cache_resource
def _warmup_model() -> None:
    """Carga el modelo de producción al inicio para que no haya espera en la pestaña ML."""
    try:
        from src.ml.predict import load_production_model

        load_production_model()
    except Exception:
        pass  # Si el modelo no está registrado, se mostrará el error en la página


def main() -> None:
    """Punto de entrada principal del dashboard."""
    st.markdown(CSS, unsafe_allow_html=True)
    _inject_sidebar_toggle()
    _warmup_model()

    with st.sidebar:
        st.markdown(
            f"""<div style="padding:14px 6px 18px 6px;">
                    <div style="font-family:{_FONT_MONO}; font-size:12px; letter-spacing:0.06em;
                                color:{_COLOR_DIM}; text-transform:uppercase; margin-bottom:2px;">
                        // Chicago Crimes
                    </div>
                    <div style="font-family:{_FONT_MONO}; font-weight:700; font-size:16px; color:{_COLOR_TEXT};">
                        Analytics Suite
                    </div>
                </div>""",
            unsafe_allow_html=True,
        )
        selected = option_menu(
            menu_title=None,
            options=["Analytics", "Predicción ML"],
            icons=["bar-chart-line", "cpu"],
            default_index=0,
            styles={
                "container": {"background-color": "transparent", "padding": "0"},
                "icon": {"color": _COLOR_MUTED, "font-size": "15px"},
                "nav-link": {
                    "font-family": _FONT_SANS,
                    "font-size": "14px",
                    "color": _COLOR_MUTED,
                    "border-radius": "8px",
                    "margin": "3px 0",
                    "padding": "9px 12px",
                },
                "nav-link-selected": {
                    "background-color": "#182137",
                    "color": _COLOR_TEXT,
                    "font-weight": "600",
                    "border-left": f"3px solid {_COLOR_CRIME}",
                },
            },
        )
        st.markdown(
            f"""<div style="position:fixed; bottom:16px; font-size:11.5px; color:{_COLOR_DIM};">
                    8,464,732 registros<br>2001–2026 · Chicago, IL
                </div>""",
            unsafe_allow_html=True,
        )

    manager = _get_manager()

    if selected == "Analytics":
        analytics.render(manager)
    else:
        prediction.render(manager)


if __name__ == "__main__":
    main()
