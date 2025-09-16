import cv2
import numpy as np
import torch

def prepare_image(image_path, device, img_size=640):
    """Loads and preprocesses an image for YOLO inference."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    return img_rgb, img_tensor