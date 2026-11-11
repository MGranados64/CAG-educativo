"""
app.py — La puerta de entrada de la app educativa (Streamlit)
==============================================================

Streamlit convierte scripts de Python en páginas web. Este archivo solo arma
el "índice" de la app; cada página vive en su propio archivo:

  🧪 Laboratorio  → paginas/laboratorio.py  (los tres paneles para experimentar)
  ⚔️ CAG vs RAG   → paginas/cara_a_cara.py  (los dos sistemas, cara a cara)
  📖 Teoría       → paginas/teoria.py       (muestra TEORIA.md)

Ejecuta:   streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="CAG Educativo", page_icon="🧠", layout="wide")

navegacion = st.navigation(
    [
        st.Page("paginas/laboratorio.py", title="Laboratorio", icon="🧪", default=True),
        st.Page("paginas/cara_a_cara.py", title="CAG vs RAG", icon="⚔️"),
        st.Page("paginas/teoria.py", title="Teoría", icon="📖"),
    ]
)
navegacion.run()
