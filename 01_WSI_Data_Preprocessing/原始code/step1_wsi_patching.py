import os

DLL_PATH = r"D:\openslide-win64-20171122\bin"
if os.path.exists(DLL_PATH) and hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(DLL_PATH)
    except Exception:
        pass

import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm

PATHS = {
    r'D:\sjd\project\liver_failure\match\Liver biopsy': r'D:\sjd\project\liver_failure\match\Patches_Raw\Liver biopsy',
    r'D:\sjd\project\liver_failure\match\Liver transplantation': r'D:\sjd\project\liver_failure\match\Patches_Raw\Liver transplantation'
}

PATCH_SIZE = 512

def process_slide(slide_path, slide_name, output_root):
    save_dir = os.path.join(output_root, slide_name)
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        return

    os.makedirs(save_dir, exist_ok=True)
    
    try:
        slide = openslide.OpenSlide(slide_path)
        bounds_x = int(slide.properties.get('openslide.bounds-x', 0))
        bounds_y = int(slide.properties.get('openslide.bounds-y', 0))
        bounds_w = int(slide.properties.get('openslide.bounds-width', slide.dimensions[0]))
        bounds_h = int(slide.properties.get('openslide.bounds-height', slide.dimensions[1]))

        patch_coords = []
        for y in range(0, bounds_h, PATCH_SIZE):
            for x in range(0, bounds_w, PATCH_SIZE):
                patch_coords.append((x + bounds_x, y + bounds_y))
        
        count = 0
        for x, y in tqdm(patch_coords, desc=slide_name, leave=False):
            w = min(PATCH_SIZE, bounds_w - (x - bounds_x))
            h = min(PATCH_SIZE, bounds_h - (y - bounds_y))
            if w < PATCH_SIZE // 2 or h < PATCH_SIZE // 2: 
                continue

            patch_rgba = slide.read_region((x, y), 0, (w, h))
            patch_rgb = np.array(patch_rgba)[:, :, :3]

            if np.mean(patch_rgb) > 230 or np.std(patch_rgb) < 15:
                continue

            img = Image.fromarray(patch_rgb)
            save_name = f"{slide_name}_x{x}_y{y}.jpg"
            img.save(os.path.join(save_dir, save_name), quality=85)
            count += 1
            
        slide.close()
    except Exception:
        pass

def main():
    for source_dir, output_root in PATHS.items():
        if not os.path.exists(source_dir):
            continue
            
        os.makedirs(output_root, exist_ok=True)
        wsi_files = []
        
        for root, dirs, files in os.walk(source_dir):
            for f in files:
                if f.lower().endswith(('.mrxs', '.ndpi', '.svs', '.tif', '.tiff')):
                    wsi_files.append(os.path.join(root, f))
        
        for file_path in tqdm(wsi_files, desc=f"Processing {os.path.basename(source_dir)}"):
            slide_name = os.path.splitext(os.path.basename(file_path))[0]
            process_slide(file_path, slide_name, output_root)

if __name__ == '__main__':
    main()