# main_app.py
import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Dashboard de Explicabilidad YOLO",
    page_icon="🤖"
)

st.title("Bienvenido al Dashboard Interactivo de Explicabilidad para YOLO")
st.markdown("""
Esta herramienta te permite analizar y comparar el rendimiento de diferentes modelos YOLO (v7, v8, etc.).

**Instrucciones:**
1.  Navega al **'⚙️ Panel de Control'** en la barra lateral para seleccionar un modelo, una imagen y otros parámetros.
2.  Una vez configurado, ve al **'📊 Dashboard Visual'** para ver los resultados.
""")