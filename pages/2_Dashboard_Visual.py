# pages/2_📊_Dashboard_Visual.py
import os
import streamlit as st
from PIL import Image

from utils.model_loader import (load_model, get_predictions,
                                draw_prediction_boxes, draw_ground_truth_boxes)
from utils.processing import prepare_image
from utils.visualization import generate_feature_maps_images, generate_grad_cam_image

st.set_page_config(page_title="Dashboard Visual", layout="wide")

# Ya no necesitamos inyectar CSS aquí. El archivo config.toml lo maneja todo.

# Notificaciones
if 'notifications' in st.session_state and st.session_state.notifications:
    for n in st.session_state.notifications: st.toast(n, icon="🎉")
    st.session_state.notifications = []

if 'config' not in st.session_state or not st.session_state['config']:
    st.warning("Por favor, ve al '⚙️ Panel de Control' para configurar un análisis primero.")
    st.stop()

# --- Título y Carga ---
config = st.session_state['config']
st.title("Dashboard de Explicabilidad")
st.caption(
    f"Modelo: **{config['model_name']}** | Imagen: **{config['image_name']}** | Confianza: **{config['conf_threshold']}**")
st.markdown("---")


@st.cache_data
def generate_all_visuals(config):
    image_path = f"{config['dataset_path']}/images/{config['image_name']}"
    label_path = f"{config['dataset_path']}/labels/{os.path.splitext(config['image_name'])[0]}.txt"

    model, device = load_model(config['model_name'], config['weights_path'], config['type'])
    img_rgb, img_tensor, geometry_info = prepare_image(image_path, device)
    predictions = get_predictions(model, config['type'], img_tensor, conf_thres=config['conf_threshold'])
    tensor_shape = img_tensor.shape[2:]

    content_list = []
    content_list.append(
        ("Realidad (Ground Truth)", draw_ground_truth_boxes(img_rgb, label_path, config['class_names'])))

    num_dets = len(predictions[0]) if predictions and predictions[0] is not None else 0
    content_list.append((f"Predicción ({num_dets} Detecciones)",
                         draw_prediction_boxes(img_rgb, predictions, tensor_shape, config['class_names'])))

    if config.get('show_grad_cam'):
        grad_cam_result = generate_grad_cam_image(model, img_rgb, img_tensor, geometry_info, 75)
        if grad_cam_result: content_list.append(grad_cam_result)

    try:
        content_list.append(("Arquitectura YOLOv7", Image.open("Images/yolo7_arq.webp")))
    except FileNotFoundError:
        pass

    if config.get('show_feature_maps'):
        content_list.extend(
            generate_feature_maps_images(model, img_rgb, img_tensor, geometry_info, config['feature_layers']))

    return content_list


grid_content = generate_all_visuals(config)

# --- Creación de la Cuadrícula 4x3 ---
COLS, ROWS = 4, 3
for i in range(ROWS):
    cols = st.columns(COLS)
    for j in range(COLS):
        item_index = i * COLS + j
        if item_index < len(grid_content):
            label, image = grid_content[item_index]
            with cols[j]:
                st.image(image, use_container_width=True)
                st.caption(label)