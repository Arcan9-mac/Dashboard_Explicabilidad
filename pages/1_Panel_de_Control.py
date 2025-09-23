# pages/1_⚙️_Panel_de_Control.py
import streamlit as st
import os
import yaml
from datetime import datetime

st.set_page_config(
    page_title="⚙️ Panel de Control",
    layout="wide"
)


@st.cache_data
def load_models_config(config_path='configs/models_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# --- Inicializar Session State ---
if 'config' not in st.session_state:
    st.session_state['config'] = {}
if 'log_messages' not in st.session_state:
    st.session_state['log_messages'] = []
if 'notifications' not in st.session_state:
    st.session_state['notifications'] = []


def log_message(message):
    """Añade un mensaje con timestamp al log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_messages.append(f"[{timestamp}] {message}")


# --- Configuración de la Página ---
st.title("⚙️ Panel de Control")
st.markdown("Selecciona el modelo y los parámetros para el análisis.")

models_config = load_models_config()
model_names = list(models_config.keys())

# --- Widgets de Selección ---
st.subheader("1. Selección de Modelo e Imagen")
selected_model_name = st.selectbox("Modelo a evaluar:", model_names)

DATASET_PATH = models_config[selected_model_name]['dataset_path']
image_folder = os.path.join(DATASET_PATH, "images")
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
selected_image_file = st.selectbox("Imagen de validación:", image_files)

st.subheader("2. Parámetros de Inferencia y Visualización")
conf_threshold = st.slider("Umbral de Confianza:", 0.0, 1.0, 0.25, 0.01)
show_feature_maps = st.checkbox("Mostrar Mapas de Características", value=True)
show_grad_cam = st.checkbox("Mostrar Grad-CAM", value=False)

# --- Botón para aplicar la configuración ---
if st.button("✅ Aplicar Configuración y Enviar al Dashboard"):
    st.session_state['config'] = {
        'model_name': selected_model_name,
        'image_name': selected_image_file,
        'conf_threshold': conf_threshold,
        'show_feature_maps': show_feature_maps,
        'show_grad_cam': show_grad_cam,
        **models_config[selected_model_name]
    }

    # Limpiar log anterior y añadir nuevos mensajes
    st.session_state.log_messages = []
    log_message("Configuración aplicada con éxito.")
    log_message(f"Modelo: {selected_model_name}")
    log_message(f"Imagen: {selected_image_file}")
    log_message(f"Confianza: {conf_threshold}")

    # Enviar notificación al dashboard
    st.session_state.notifications.append("✅ ¡Nueva configuración recibida! El dashboard se ha actualizado.")

    st.success("¡Configuración guardada!")

# --- Log de Actividad ---
with st.expander("Ver Log de Actividad", expanded=True):
    if st.session_state.log_messages:
        for msg in reversed(st.session_state.log_messages):
            st.text(msg)
    else:
        st.caption("Aún no hay actividad para registrar.")