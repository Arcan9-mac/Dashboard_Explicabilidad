# utils/model_loader.py
import torch
import streamlit as st
from ultralytics import YOLO
from utils.yolov7_compat import attempt_load, select_device
from utils.general import non_max_suppression, xywh2xyxy
import numpy as np
import cv2
import torch

@st.cache_resource
def load_model(model_name, weights_path, model_type):
    device = select_device('')
    print(f"Cargando modelo '{model_name}' de tipo '{model_type}'...")

    if model_type == 'yolov7':
        model = attempt_load(weights_path, map_location=device)
        model.eval()
        for module in model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False
        return model, device

    elif model_type == 'yolov8' or model_type == 'yolov11': # Asumiendo que v11 usa la misma API
        model = YOLO(weights_path)
        return model, device

    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")

def get_predictions(model, img_tensor):
    """Obtains predictions from the model."""
    with torch.no_grad():
        pred = model(img_tensor, augment=False)[0]
    return non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

def draw_prediction_boxes(image, predictions, class_names):
    """Draws prediction boxes (in green)."""
    img_with_boxes = image.copy()
    if predictions[0] is not None:
        for *xyxy, conf, cls in reversed(predictions[0]):
            label = f'{class_names[int(cls)]} {conf:.2f}'
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_with_boxes

def draw_ground_truth_boxes(image, label_path, class_names):
    """Reads a YOLO label file and draws ground truth boxes (in red)."""
    img_with_gt = image.copy()
    h, w, _ = image.shape
    import os
    import streamlit as st
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                label = class_names[cls_id]
                cv2.rectangle(img_with_gt, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_with_gt, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    except FileNotFoundError:
        st.warning(f"Label file not found: {os.path.basename(label_path)}")
    return img_with_gt