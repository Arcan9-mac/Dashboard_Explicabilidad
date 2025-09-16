# pages/1_⚙️_Panel_de_Control.py
import streamlit as st
import os
import yaml

# Cargar la configuración de los modelos desde el archivo YAML
@st.cache_data
def load_models_config(config_path='configs/models_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Configuración de la Página ---
st.title("⚙️ Panel de Control")
st.write("Selecciona el modelo y los parámetros para el análisis.")

models_config = load_models_config()
model_names = list(models_config.keys())

# --- Inicializar Session State si no existe ---
if 'config' not in st.session_state:
    st.session_state['config'] = {}

# --- Widgets de Selección ---
st.header("1. Selección de Modelo e Imagen")

selected_model_name = st.selectbox("Selecciona el modelo a evaluar:", model_names)

# Cargar dinámicamente las imágenes del dataset asociado al modelo
DATASET_PATH = models_config[selected_model_name]['dataset_path']
image_folder = os.path.join(DATASET_PATH, "images")
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
selected_image_file = st.selectbox("Selecciona una imagen de validación:", image_files)

st.header("2. Parámetros de Visualización")
show_feature_maps = st.checkbox("Mostrar Mapas de Características", value=True)
show_grad_cam = st.checkbox("Mostrar Grad-CAM", value=False)

# (Aquí podrías añadir más opciones: umbral de confianza, capa para Grad-CAM, etc.)

# --- Botón para aplicar la configuración ---
if st.button("Aplicar Configuración y Preparar Dashboard"):
    st.session_state['config'] = {
        'model_name': selected_model_name,
        'image_name': selected_image_file,
        'show_feature_maps': show_feature_maps,
        'show_grad_cam': show_grad_cam,
        **models_config[selected_model_name] # Añade toda la config del modelo
    }
    st.success("¡Configuración guardada! Navega al 'Dashboard Visual' para ver los resultados.")
    st.info(f"Modelo seleccionado: **{selected_model_name}**")
    st.info(f"Imagen seleccionada: **{selected_image_file}**")