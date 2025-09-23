# main_app.py
import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Dashboard de Explicabilidad YOLO",
    page_icon="🤖"
)

st.title("🚀 Dashboard Interactivo de Explicabilidad para YOLO")
st.markdown("---")
st.header("Selecciona una opción para comenzar:")

col1, col2 = st.columns(2)

# Nuevos colores de la paleta opaca
button_bg_color = "#2A3A2F" # secondaryBackgroundColor
border_color = "#6A8A74"    # primaryColor

with col1:
    st.markdown(f"""
    <a href="/Panel_de_Control" target="_blank" style="
        display: block;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: {button_bg_color};
        color: white;
        text-align: center;
        text-decoration: none;
        font-size: 1.25rem;
        font-weight: bold;
        border: 1px solid {border_color};
    ">
        ⚙️ Abrir Panel de Control
    </a>
    """, unsafe_allow_html=True)
    st.info("Configura aquí el modelo, la imagen y los parámetros de visualización.")

with col2:
    st.markdown(f"""
    <a href="/Dashboard_Visual" target="_blank" style="
        display: block;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: {button_bg_color};
        color: white;
        text-align: center;
        text-decoration: none;
        font-size: 1.25rem;
        font-weight: bold;
        border: 1px solid {border_color};
    ">
        📊 Abrir Dashboard Visual
    </a>
    """, unsafe_allow_html=True)
    st.info("Visualiza y compara aquí los resultados del análisis.")