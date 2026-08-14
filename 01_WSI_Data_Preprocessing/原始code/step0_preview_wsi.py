import os
import openslide
from openslide import OpenSlide

SOURCE_DIRS = [
    r'G:\90天生存',
    r'G:\90天死亡'
]

OUTPUT_ROOT = r'D:\sjd\课题\肝衰竭\2.0 data\preview_thumbnails_all_death'
DLL_PATH = r"D:\openslide-win64-20171122\bin" 

if os.path.exists(DLL_PATH) and hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(DLL_PATH)
    except Exception as e:
        print(e)

def generate_thumbnails():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    for source_dir in SOURCE_DIRS:
        if not os.path.exists(source_dir):
            continue
            
        wsi_files = []
        for root, dirs, files in os.walk(source_dir):
            for f in files:
                if f.lower().endswith(('.mrxs', '.ndpi', '.svs', '.tif', '.tiff')):
                    wsi_files.append(os.path.join(root, f))
        
        for file_path in wsi_files:
            file_name = os.path.basename(file_path)
            save_name = f"{file_name}.jpg"
            save_path = os.path.join(OUTPUT_ROOT, save_name)
            
            if os.path.exists(save_path):
                continue

            try:
                slide = OpenSlide(file_path)
                thumb = slide.get_thumbnail((1024, 1024))
                thumb.save(save_path, quality=80)
                print(f"Generated: {save_name}")
            except Exception as e:
                print(f"Error {file_name}: {e}")

if __name__ == '__main__':
    generate_thumbnails()