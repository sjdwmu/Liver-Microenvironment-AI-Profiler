import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

DATA_ROOT = r'D:\sjd\project\liver_failure\match\Patches_Raw'
TRASH_DIR = r'D:\sjd\project\liver_failure\match\Patches_Trash'

DARK_THRESHOLD = 60
SATURATION_THRESHOLD = 10
BLUR_THRESHOLD = 150
BAD_PURPLE_H_MIN = 120
BAD_PURPLE_H_MAX = 175
BAD_PURPLE_S_MIN = 40
BAD_PURPLE_RATIO_LIMIT = 80.0
FLAT_STD_THRESHOLD = 15.0

def is_bad_image(img_path):
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img is None: 
            return True
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mean_brightness = np.mean(gray)
        if mean_brightness < DARK_THRESHOLD:
            return True
            
        std_dev = np.std(gray)
        if std_dev < FLAT_STD_THRESHOLD:
            return True
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        mean_saturation = np.mean(hsv[:, :, 1])
        if mean_saturation < SATURATION_THRESHOLD:
            return True
            
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < BLUR_THRESHOLD:
            return True
            
        tissue_mask = gray < 235
        if np.sum(tissue_mask) < 500: 
            return False 
            
        tissue_hsv = hsv[tissue_mask]
        
        hue_values = tissue_hsv[:, 0]
        sat_values = tissue_hsv[:, 1]
        
        bad_purple_mask = (hue_values >= BAD_PURPLE_H_MIN) & \
                          (hue_values <= BAD_PURPLE_H_MAX) & \
                          (sat_values >= BAD_PURPLE_S_MIN)
                          
        total_tissue_pixels = len(hue_values)
        bad_purple_pixels = np.sum(bad_purple_mask)
        
        purple_ratio = (bad_purple_pixels / total_tissue_pixels) * 100
        
        if purple_ratio > BAD_PURPLE_RATIO_LIMIT:
            return True
            
        return False
    except Exception:
        return True

def main():
    if not os.path.exists(DATA_ROOT):
        return

    files_to_check = []
    for root, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                files_to_check.append(os.path.join(root, file))

    for img_path in tqdm(files_to_check, desc="Cleaning"):
        if is_bad_image(img_path):
            rel_path = os.path.relpath(img_path, DATA_ROOT)
            trash_path = os.path.join(TRASH_DIR, rel_path)
            
            os.makedirs(os.path.dirname(trash_path), exist_ok=True)
            shutil.move(img_path, trash_path)

if __name__ == '__main__':
    main()