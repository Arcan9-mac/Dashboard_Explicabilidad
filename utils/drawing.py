import cv2
from .model_loader import get_predictions, draw_prediction_boxes, draw_ground_truth_boxes

def draw_all_boxes(model, img_rgb, img_tensor, label_path, class_names):
    """Draws both ground truth and prediction boxes on the image."""
    predictions = get_predictions(model, img_tensor)
    img_pred = draw_prediction_boxes(img_rgb, predictions, class_names)
    img_final = draw_ground_truth_boxes(img_pred, label_path, class_names)
    return img_final