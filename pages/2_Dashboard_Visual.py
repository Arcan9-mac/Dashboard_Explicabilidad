# pages/2_📊_Dashboard_Visual.py
import streamlit as st
import os


from utils.model_loader import load_model # Necesitaremos crear esta función
from utils.processing import prepare_image # Y esta
from utils.drawing import draw_all_boxes
from utils.visualization import display_feature_maps, display_grad_cam

st.title("📊 Dashboard de Análisis y Explicabilidad")

# Verificar si la configuración ha sido establecida desde el panel de control
if 'config' not in st.session_state or not st.session_state['config']:
    st.warning("Por favor, ve al 'Panel de Control' para configurar un análisis primero.")
    st.stop()

# --- Cargar configuración ---
config = st.session_state['config']
model_name = config['model_name']
image_name = config['image_name']
image_path = f"{config['dataset_path']}/images/{image_name}"
label_path = f"{config['dataset_path']}/labels/{os.path.splitext(image_name)[0]}.txt"

# --- Lógica Principal (usando las funciones de utils) ---
st.header(f"Análisis para el modelo: `{model_name}`")
st.subheader(f"Imagen: `{image_name}`")

# 1. Cargar el modelo (esta función manejará v7, v8, etc.)
model, device = load_model(model_name, config['weights_path'], config['type'])

# 2. Preparar la imagen
img_rgb, img_tensor = prepare_image(image_path, device)

# 3. Obtener predicciones y dibujar cajas
# (Esta función combinará la lógica de predicción y dibujo)
final_image = draw_all_boxes(model, img_rgb, img_tensor, label_path, config['class_names'])
st.image(final_image, caption="Realidad (Rojo) vs. Predicción (Verde)", use_container_width=True)

st.divider()

# 4. Visualizar Mapas de Características (si está activado)
if config['show_feature_maps']:
    display_feature_maps(model, img_rgb, img_tensor, config['feature_layers'])

# 5. Visualizar Grad-CAM (si está activado)
if config['show_grad_cam']:
    # La lógica de Grad-CAM es más compleja y requerirá una capa seleccionada
    # por el usuario en el panel de control
    display_grad_cam(model, img_rgb, img_tensor, 75) # target_layer vendría de la config
