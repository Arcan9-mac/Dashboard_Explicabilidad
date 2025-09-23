# utils/model_loader.py
import torch
import streamlit as st
from ultralytics import YOLO
from .yolov7_compat import attempt_load, select_device
from .general import non_max_suppression, scale_coords
import numpy as np
import cv2


@st.cache_resource
def load_model(model_name, weights_path, model_type):
    # --- MENSAJE MOVIDO A LA CONSOLA ---
    print(f"Cargando modelo '{model_name}' de tipo '{model_type}'...")
    device = select_device('')

    if model_type == 'yolov7':
        model = attempt_load(weights_path, map_location=device)
        model.eval()
        for module in model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False
        return model, device

    elif model_type in ['yolov8', 'yolov11']:
        model = YOLO(weights_path)
        # --- INICIO: PARCHE DE COMPATIBILIDAD PARA ULTRALYTICS ---
        # Se llama a .fuse() explícitamente para evitar el error en versiones antiguas.
        try:
            model.fuse()
        except Exception as e:
            # Si la fusión falla por alguna razón, solo se imprime en consola y se continúa.
            print(f"Advertencia: No se pudo fusionar el modelo YOLOv8. Error: {e}")
        # --- FIN: PARCHE DE COMPATIBILIDAD ---
        return model, device

    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")


def get_predictions(model, model_type, img_tensor, conf_thres=0.25, iou_thres=0.45):
    """
    Obtiene predicciones del modelo, manejando la diferencia entre YOLOv7 y YOLOv8.
    """
    if model_type == 'yolov7':
        detector = model.model[-1]
        num_classes_model = detector.nc
    elif model_type in ['yolov8', 'yolov11']:
        num_classes_model = len(model.names)

    print(f"DIAGNÓSTICO (Consola): El archivo de pesos del modelo está configurado para {num_classes_model} clases.")

    if model_type == 'yolov7':
        with torch.no_grad():
            pred = model(img_tensor, augment=False)[0]
        return non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

    elif model_type in ['yolov8', 'yolov11']:
        # Ahora esta llamada no debería causar el error de fuse.
        results = model.predict(img_tensor, conf=conf_thres, iou=iou_thres)
        return [results[0].boxes.data]
    else:
        raise ValueError(f"La lógica de predicción para el tipo de modelo '{model_type}' no está implementada.")


def draw_prediction_boxes(image, predictions, tensor_shape, class_names):
    """
    Dibuja cajas de predicción (en verde) escalando las coordenadas.
    """
    img_with_boxes = image.copy()
    dets = predictions[0]

    if not isinstance(class_names, list):
        st.error("Error de configuración: `class_names` debe ser una lista en tu archivo .yaml.")
        return image

    num_classes_config = len(class_names)
    print(f"DIAGNÓSTICO (Consola): Tu config provee {num_classes_config} nombres de clase.")

    if dets is not None and len(dets):
        dets[:, :4] = scale_coords(tensor_shape, dets[:, :4], image.shape).round()

        for *xyxy, conf, cls in reversed(dets):
            class_id = int(cls)
            if class_id < num_classes_config:
                label = f'{class_names[class_id]} {conf:.2f}'
                color = (0, 255, 0)
            else:
                label = f'ID Inesperado: {class_id} ({conf:.2f})'
                color = (0, 255, 255)

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img_with_boxes


def draw_ground_truth_boxes(image, label_path, class_names):
    """Dibuja cajas de realidad (en rojo)."""
    img_with_gt = image.copy()
    h, w, _ = image.shape
    import os
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                if cls_id < len(class_names):
                    label = class_names[cls_id]
                    x_center, y_center, width, height = map(float, parts[1:])
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    cv2.rectangle(img_with_gt, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img_with_gt, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    except FileNotFoundError:
        st.warning(f"Archivo de etiqueta no encontrado: {os.path.basename(label_path)}")
    return img_with_gt