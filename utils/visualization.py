# utils/visualization.py
import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    st.error("Por favor, instala pytorch-grad-cam: pip install grad-cam")


# --- NOMBRE DE FUNCIÓN CORREGIDO ---
def generate_feature_maps_images(model, img_rgb, img_tensor, geometry_info, feature_layers_config):
    """
    Genera las imágenes de los mapas de características y las devuelve en una lista.
    No usa st.image, solo procesa y devuelve los resultados.
    """
    _, pad = geometry_info
    pad_w, pad_h = int(pad[0]), int(pad[1])
    tensor_h, tensor_w = img_tensor.shape[2:]

    all_modules = list(model.modules())
    hooks, features = [], {}
    output_images = []

    def get_features_hook(name):
        def hook(model, input, output): features[name] = output.cpu().detach()

        return hook

    for alias, idx in feature_layers_config.items():
        if 0 <= idx < len(all_modules):
            hooks.append(all_modules[idx].register_forward_hook(get_features_hook(alias)))

    if not hooks: return []

    with torch.no_grad():
        model(img_tensor)
    for hook in hooks: hook.remove()

    if not features: return []

    for alias, fmap in features.items():
        label = f"Mapa: {alias.replace('_', ' ').title()}"
        mean_fmap = torch.mean(fmap.squeeze(), 0).numpy()

        fmap_upscaled = cv2.resize(mean_fmap, (tensor_w, tensor_h))

        h_unpad, w_unpad = tensor_h - 2 * pad_h, tensor_w - 2 * pad_w
        fmap_cropped = fmap_upscaled[pad_h: pad_h + h_unpad, pad_w: pad_w + w_unpad]

        if fmap_cropped.size == 0: continue

        heatmap = cv2.normalize(fmap_cropped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        heatmap_resized = cv2.resize(heatmap_rgb, (img_rgb.shape[1], img_rgb.shape[0]))

        overlay_img = cv2.addWeighted(img_rgb, 0.6, heatmap_resized, 0.4, 0)
        output_images.append((label, overlay_img))

    return output_images


class YOLOv7ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(YOLOv7ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def generate_grad_cam_image(model, img_rgb, img_tensor, geometry_info, target_layer_identifier):
    """
    Genera la imagen de Grad-CAM y la devuelve. No usa st.image.
    """
    _, pad = geometry_info
    pad_w, pad_h = int(pad[0]), int(pad[1])
    tensor_h, tensor_w = img_tensor.shape[2:]

    all_modules = list(model.modules())
    if not (0 <= target_layer_identifier < len(all_modules)):
        return None

    target_module = all_modules[target_layer_identifier]
    wrapped_model = YOLOv7ModelWrapper(model)
    cam = EigenCAM(model=wrapped_model, target_layers=[target_module])

    grayscale_cam = cam(input_tensor=img_tensor)[0, :]

    h_unpad, w_unpad = tensor_h - 2 * pad_h, tensor_w - 2 * pad_w
    cam_cropped = grayscale_cam[pad_h: pad_h + h_unpad, pad_w: pad_w + w_unpad]

    if cam_cropped.size == 0: return None

    cam_resized_to_original = cv2.resize(cam_cropped, (img_rgb.shape[1], img_rgb.shape[0]))
    img_float = np.float32(img_rgb) / 255
    cam_image = show_cam_on_image(img_float, cam_resized_to_original, use_rgb=True)

    return ("Análisis Grad-CAM", cam_image)