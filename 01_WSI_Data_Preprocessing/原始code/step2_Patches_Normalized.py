import os
import cv2
import numpy as np
from tqdm import tqdm

SOURCE_DIR = r'D:\sjd\project\liver_failure\match\Patches_Raw'
TARGET_IMG_PATH = r'D:\sjd\project\liver_failure\match\Patches_Raw\53879 (12)_x4096_y60252.jpg'
OUTPUT_DIR = r'D:\sjd\project\liver_failure\match\Patches_Normalized'

def cv_imread(file_path):
    try:
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        return img
    except:
        return None

def cv_imwrite(file_path, img):
    try:
        cv2.imencode('.jpg', img)[1].tofile(file_path)
    except:
        pass

def get_lab_stats(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray < 235).astype(np.uint8)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    means, stds = [], []
    for i in range(3):
        channel = lab[:, :, i]
        tissue_pixels = channel[mask > 0]
        if len(tissue_pixels) > 100:
            means.append(np.mean(tissue_pixels))
            stds.append(np.std(tissue_pixels) + 1e-6)
        else:
            means.append(np.mean(channel))
            stds.append(np.std(channel) + 1e-6)
    return np.array(means), np.array(stds)

def reinhard_normalize(src_rgb, target_means, target_stds):
    src_means, src_stds = get_lab_stats(src_rgb)
    lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    for i in range(3):
        lab[:, :, i] = ((lab[:, :, i] - src_means[i]) * (target_stds[i] / src_stds[i])) + target_means[i]
        
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    target_img = cv_imread(TARGET_IMG_PATH)
    if target_img is None:
        return
    target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    target_means, target_stds = get_lab_stats(target_img_rgb)
    
    files_to_process = []
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                files_to_process.append(os.path.join(root, file))
                
    for img_path in tqdm(files_to_process, desc="Normalizing"):
        rel_path = os.path.relpath(img_path, SOURCE_DIR)
        save_path = os.path.join(OUTPUT_DIR, rel_path)
        
        if os.path.exists(save_path) and os.path.getsize(save_path) > 100:
            continue
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        img = cv_imread(img_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.sum(gray < 235) == 0:
            continue
            
        try:
            norm_rgb = reinhard_normalize(img_rgb, target_means, target_stds)
            norm_bgr = cv2.cvtColor(norm_rgb, cv2.COLOR_RGB2BGR)
            cv_imwrite(save_path, norm_bgr)
        except:
            continue

if __name__ == '__main__':
    main()