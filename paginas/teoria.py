"""
paginas/teoria.py — La teoría de CAG, ligera y con fuentes.

El texto vive en TEORIA.md (en la raíz del proyecto) para poder leerlo igual de
bien aquí que en GitHub. Esta página solo lo muestra.
"""

from pathlib import Path

import streamlit as st

RUTA_TEORIA = Path(__file__).parent.parent / "TEORIA.md"

st.page_link("paginas/laboratorio.py", label="Ir al laboratorio para experimentar", icon="🧪")
st.markdown(RUTA_TEORIA.read_text("utf-8"))
st.divider()
st.page_link("paginas/laboratorio.py", label="¿Listo? Ve al laboratorio y compruébalo tú mismo", icon="🧪")
