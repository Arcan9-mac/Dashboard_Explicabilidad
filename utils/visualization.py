import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image # <-- ESTA ES LA LÍNEA QUE FALTABA
except ImportError:
    st.error("Por favor, instala pytorch-grad-cam: pip install grad-cam")

# --- FUNCIÓN CORREGIDA PARA MAPAS DE CARACTERÍSTICAS (USANDO ÍNDICES) ---
def display_feature_maps(model, img_rgb, img_tensor, feature_layers_config):
    """
    Muestra los mapas de características superpuestos para las capas seleccionadas por ÍNDICE.
    'feature_layers_config' es un diccionario como {'backbone': 10, 'neck': 50}
    """
    st.header("Visualización de Mapas de Características")

    # Obtener la lista completa de módulos una sola vez
    all_modules = list(model.modules())

    hooks = []
    features = {}

    def get_features_hook(name):
        def hook(model, input, output):
            features[name] = output.cpu().detach()

        return hook

    # Registrar hooks usando los ÍNDICES de las capas
    for layer_alias, layer_idx in feature_layers_config.items():
        # Validar que el índice esté dentro del rango del modelo
        if 0 <= layer_idx < len(all_modules):
            layer_module = all_modules[layer_idx]
            hooks.append(layer_module.register_forward_hook(get_features_hook(layer_alias)))
        else:
            st.warning(f"Índice de capa fuera de rango: {layer_idx}. Se omitirá.")

    # Si no se pudo registrar ningún hook, detener la ejecución para evitar el error de st.columns
    if not hooks:
        st.error(
            "No se pudo registrar ningún hook para las capas de características especificadas. Revisa los índices en tu configuración.")
        return  # Salir de la función

    # Forward pass para activar los hooks
    with torch.no_grad():
        model(img_tensor)

    # Limpiar hooks inmediatamente después de usarlos
    for hook in hooks:
        hook.remove()

    # Visualizar los mapas de características capturados
    # Validar que 'features' no esté vacío antes de crear las columnas
    if not features:
        st.warning(
            "No se capturaron mapas de características. El forward pass puede haber fallado o los hooks no se activaron.")
        return

    cols = st.columns(len(features))
    for i, (layer_alias, fmap) in enumerate(features.items()):
        with cols[i]:
            # Usamos el alias ('backbone', 'neck') que es más descriptivo
            st.write(f"**{layer_alias.replace('_', ' ').title()}** (Capa {feature_layers_config[layer_alias]})")

            # Crear un heatmap promediando los canales
            mean_fmap = torch.mean(fmap.squeeze(), 0).numpy()

            heatmap = cv2.normalize(mean_fmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            heatmap_resized = cv2.resize(heatmap_rgb, (img_rgb.shape[1], img_rgb.shape[0]))

            overlay_img = cv2.addWeighted(img_rgb, 0.6, heatmap_resized, 0.4, 0)
            st.image(overlay_img, caption=f"Activación media de la capa", use_container_width=True)


# --- INICIO DE LA SOLUCIÓN: EL ADAPTADOR ---
# Esta clase envuelve el modelo YOLOv7 para asegurar que siempre devuelva un solo tensor.
class YOLOv7ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(YOLOv7ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Llama al modelo original
        outputs = self.model(x)

        # Si la salida es una tupla o lista (comportamiento en modo de entrenamiento o complejo),
        # devuelve solo el primer elemento, que es la salida de detección principal.
        if isinstance(outputs, (list, tuple)):
            return outputs[0]

        # Si ya es un tensor, devuélvelo directamente.
        return outputs


# Pega esto en tu archivo utils/visualization.py, reemplazando la función display_grad_cam.
# El resto del archivo (imports, wrapper, display_feature_maps) no cambia.

def display_grad_cam(model, img_rgb, img_tensor, target_layer_identifier):
    """
    Muestra el heatmap de Grad-CAM usando un 'wrapper' para asegurar la compatibilidad del modelo.
    """
    st.header("Visualización con Grad-CAM")

    target_module = None
    if isinstance(target_layer_identifier, int):
        st.info(f"Generando Grad-CAM para la capa con índice: `{target_layer_identifier}`")
        all_modules = list(model.modules())
        if 0 <= target_layer_identifier < len(all_modules):
            target_module = all_modules[target_layer_identifier]
        else:
            st.error(f"Error: El índice de capa '{target_layer_identifier}' está fuera de rango.")
            return
    elif isinstance(target_layer_identifier, str):
        st.info(f"Generando Grad-CAM para la capa con nombre: `{target_layer_identifier}`")
        modules_dict = dict(model.named_modules())
        target_module = modules_dict.get(target_layer_identifier)
    if target_module is None:
        st.error(f"Error: No se pudo encontrar la capa '{target_layer_identifier}' en el modelo.")
        return


    wrapped_model = YOLOv7ModelWrapper(model)
    cam = EigenCAM(model=wrapped_model, target_layers=[target_module])

    # Se genera el mapa de calor. Tendrá las dimensiones del input_tensor (ej: 640x640)
    grayscale_cam = cam(input_tensor=img_tensor)[0, :]

    # --- INICIO DE LA SOLUCIÓN ---
    # Aseguramos que la imagen a visualizar (img_rgb) tenga el mismo tamaño que el mapa de calor.
    # Obtenemos las dimensiones (alto, ancho) del mapa de calor.
    target_height, target_width = grayscale_cam.shape

    # Redimensionamos la imagen RGB a esas dimensiones usando OpenCV.
    img_rgb_resized = cv2.resize(img_rgb, (target_width, target_height))

    # Ahora usamos la imagen redimensionada para la superposición.
    img_float = np.float32(img_rgb_resized) / 255
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    st.image(cam_image, caption=f"EigenCAM para la capa {target_layer_identifier}", use_container_width=True)