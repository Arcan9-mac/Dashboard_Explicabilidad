# utils/processing.py
import cv2
import numpy as np
import torch
from .datasets import letterbox


def prepare_image(image_path, device, img_size=640):
    """
    Carga y preprocesa una imagen, devolviendo la imagen original, el tensor,
    y la información de geometría (ratio, padding) para la correcta visualización.
    """
    img0 = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    # Letterbox devuelve la imagen padeada, el ratio y los valores de padding
    img_padded, ratio, pad = letterbox(img_rgb, new_shape=img_size, auto=False, scaleup=False)

    # Convertir a formato de tensor
    img_tensor = img_padded.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_tensor = np.ascontiguousarray(img_tensor)
    img_tensor = torch.from_numpy(img_tensor).to(device).float()
    img_tensor /= 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Devolvemos la geometría para poder recortar los heatmaps más adelante
    geometry_info = (ratio, pad)
    return img_rgb, img_tensor, geometry_info