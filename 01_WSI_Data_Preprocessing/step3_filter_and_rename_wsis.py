import os
import shutil
import pandas as pd
from tqdm import tqdm

THUMBNAIL_DIR = r'./data/preview_thumbnails'
SRC_SURVIVAL_DIR = r'./data/matched_wsi/LT-free group'
SRC_DEATH_DIR = r'./data/matched_wsi/LT-group'

SELECTED_SURVIVAL_OUT = r'./data/Selected_WSIs/LT-free group'
SELECTED_DEATH_OUT = r'./data/Selected_WSIs/LT-group'

def main():
    os.makedirs(SELECTED_SURVIVAL_OUT, exist_ok=True)
    os.makedirs(SELECTED_DEATH_OUT, exist_ok=True)

    remaining_thumbnails = set()
    if os.path.exists(THUMBNAIL_DIR):
        for f in os.listdir(THUMBNAIL_DIR):
            if f.lower().endswith('.jpg'):
                orig_name = f[:-4].strip()
                orig_name = os.path.splitext(orig_name)[0]
                remaining_thumbnails.add(orig_name)

    def process_group(src_dir, out_dir):
        if not os.path.exists(src_dir):
            return
            
        wsi_dict = {}
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith(('.mrxs', '.ndpi', '.svs', '.tif', '.tiff')):
                    base_name = os.path.splitext(file)[0].strip()
                    wsi_dict[base_name] = os.path.join(root, file)

        for name in tqdm(remaining_thumbnails, desc=f"Filtering to {os.path.basename(out_dir)}"):
            if name in wsi_dict:
                src_path = wsi_dict[name]
                dst_path = os.path.join(out_dir, os.path.basename(src_path))
                
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    
                if src_path.lower().endswith('.mrxs'):
                    src_folder = os.path.splitext(src_path)[0]
                    dst_folder = os.path.splitext(dst_path)[0]
                    if os.path.exists(src_folder) and not os.path.exists(dst_folder):
                        shutil.copytree(src_folder, dst_folder)

    process_group(SRC_SURVIVAL_DIR, SELECTED_SURVIVAL_OUT)
    process_group(SRC_DEATH_DIR, SELECTED_DEATH_OUT)

if __name__ == '__main__':
    main()