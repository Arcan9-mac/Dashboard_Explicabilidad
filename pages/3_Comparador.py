# pages/3_🖼️_Comparador.py
import streamlit as st

st.set_page_config(page_title="Comparador Visual", layout="wide")

st.title("🖼️ Comparador Visual")
st.markdown("---")

# Verificar si las visualizaciones se han generado en el Dashboard
if 'grid_content' not in st.session_state or not st.session_state['grid_content']:
    st.warning("Por favor, ve al '📊 Dashboard Visual' y genera las imágenes primero.")
    st.info("Para ello, asegúrate de haber aplicado una configuración en el '⚙️ Panel de Control'.")
    st.stop()

# --- Preparación de Datos ---
# Creamos un diccionario para acceder fácilmente a las imágenes por su etiqueta
image_dict = {label: img for label, img in st.session_state['grid_content']}
image_options = list(image_dict.keys())

# --- Selección de Layout ---
st.subheader("1. Selecciona el formato de comparación")
layout = st.radio(
    "Elige el número de imágenes a comparar:",
    ('2x1 (2 imágenes)', '2x2 (4 imágenes)'),
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("---")

# --- Lógica de Visualización ---
if layout == '2x1 (2 imágenes)':
    st.subheader("2. Elige las imágenes para comparar")
    col1, col2 = st.columns(2)

    with col1:
        selection1 = st.selectbox("Imagen Izquierda:", image_options, index=0, key="sel1")
        if selection1:
            st.image(image_dict[selection1], caption=selection1, use_container_width=True)

    with col2:
        # Preseleccionar la segunda imagen si hay más de una disponible
        default_index_2 = 1 if len(image_options) > 1 else 0
        selection2 = st.selectbox("Imagen Derecha:", image_options, index=default_index_2, key="sel2")
        if selection2:
            st.image(image_dict[selection2], caption=selection2, use_container_width=True)

elif layout == '2x2 (4 imágenes)':
    st.subheader("2. Elige las imágenes para comparar")

    # Fila 1
    col1, col2 = st.columns(2)
    with col1:
        selection1 = st.selectbox("Imagen Superior Izquierda:", image_options, index=0, key="sel_tl")
        if selection1:
            st.image(image_dict[selection1], caption=selection1, use_container_width=True)

    with col2:
        default_index_2 = 1 if len(image_options) > 1 else 0
        selection2 = st.selectbox("Imagen Superior Derecha:", image_options, index=default_index_2, key="sel_tr")
        if selection2:
            st.image(image_dict[selection2], caption=selection2, use_container_width=True)

    st.markdown("---")  # Separador visual

    # Fila 2
    col3, col4 = st.columns(2)
    with col3:
        default_index_3 = 2 if len(image_options) > 2 else 0
        selection3 = st.selectbox("Imagen Inferior Izquierda:", image_options, index=default_index_3, key="sel_bl")
        if selection3:
            st.image(image_dict[selection3], caption=selection3, use_container_width=True)

    with col4:
        default_index_4 = 3 if len(image_options) > 3 else 0
        selection4 = st.selectbox("Imagen Inferior Derecha:", image_options, index=default_index_4, key="sel_br")
        if selection4:
            st.image(image_dict[selection4], caption=selection4, use_container_width=True)