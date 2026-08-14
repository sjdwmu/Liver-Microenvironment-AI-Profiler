import os
import glob
import torch
import numpy as np
import pandas as pd
import cv2
import re
import collections
import torch.nn.functional as F
import gc
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import snapshot_download
import sys
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
try:
    import archs
except ModuleNotFoundError:
    pass

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

CHANNEL_LIST = [
    'DAPI', 'TRITC', 'Cy5', 'PD-1', 'CD14', 'CD4', 'T-bet', 'CD34', 
    'CD68', 'CD16', 'CD11c', 'CD138', 'CD20', 'CD3', 'CD8', 'PD-L1',
    'CK', 'Ki67', 'Tryptase', 'Actin-D', 'Caspase3-D', 'PHH3-B', 'Transgelin'
]

VALID_CHANNELS = [ch for ch in CHANNEL_LIST if ch not in ['TRITC', 'Cy5']]
VALID_INDICES = [CHANNEL_LIST.index(ch) for ch in VALID_CHANNELS]

COLOR_MAP = {
    'DAPI': (0, 0, 255), 'PD-1': (255, 128, 0), 'CD14': (128, 128, 128),
    'CD4': (255, 192, 203), 'T-bet': (128, 0, 128), 'CD34': (165, 42, 42),
    'CD68': (255, 0, 255), 'CD16': (0, 128, 128), 'CD11c': (128, 128, 0),
    'CD138': (0, 255, 128), 'CD20': (0, 128, 255), 'CD3': (0, 255, 0),
    'CD8': (0, 255, 255), 'PD-L1': (255, 215, 0), 'CK': (255, 0, 0),
    'Ki67': (255, 255, 0), 'Tryptase': (250, 128, 114), 'Actin-D': (210, 105, 30),
    'Caspase3-D': (173, 255, 47), 'PHH3-B': (138, 43, 226), 'Transgelin': (255, 140, 0)
}

def load_gigatime_model(device):
    repo_id = "prov-gigatime/GigaTIME"
    local_dir = snapshot_download(repo_id=repo_id)
    weights_path = os.path.join(local_dir, "model.pth")
    model = archs.__dict__['gigatime'](num_classes=23, input_channels=3)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    model.to(device)
    return model

class PatchDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            return self.transform(img), img_path
        except Exception:
            blank_img = Image.new('RGB', (512, 512), (255, 255, 255))
            return self.transform(blank_img), img_path

def get_step(coords, default_step):
    if len(coords) < 2: return default_step
    diffs = [coords[i+1] - coords[i] for i in range(len(coords)-1)]
    counts = collections.Counter(diffs)
    if 0 in counts: del counts[0]
    if not counts: return default_step
    return counts.most_common(1)[0][0]

def process_wsi(wsi_id, patch_paths, model, device, target_channels, output_vis_dir, batch_size=4):
    dataset = PatchDataset(patch_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    wsi_features = []
    patch_results = []
    coord_pattern = re.compile(r'_x(\d+)_y(\d+)')
    pad_size = 32
    
    with torch.no_grad():
        for batch_imgs, paths in tqdm(dataloader, desc=f"WSI: {wsi_id}", leave=False):
            batch_imgs = batch_imgs.to(device, non_blocking=False)
            padded_imgs = F.pad(batch_imgs, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
            
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = torch.sigmoid(model(padded_imgs))
            else:
                outputs = torch.sigmoid(model(padded_imgs))
            
            outputs = outputs[:, :, pad_size:-pad_size, pad_size:-pad_size]
            
            batch_means = outputs.mean(dim=(2, 3)).cpu().numpy()
            wsi_features.extend(batch_means[:, VALID_INDICES])
            
            if target_channels:
                outputs_np = outputs.cpu().numpy()
                for i, path in enumerate(paths):
                    coord_match = coord_pattern.search(os.path.basename(path))
                    if coord_match:
                        x, y = int(coord_match.group(1)), int(coord_match.group(2))
                    else:
                        continue
                    
                    spatial_maps = {}
                    for ch in target_channels:
                        ch_idx = CHANNEL_LIST.index(ch)
                        act_map = outputs_np[i, ch_idx]
                        act_map = np.clip(act_map / 0.8, 0, 1)
                        act_map[act_map < 0.05] = 0
                        spatial_maps[ch] = (act_map * 255).astype(np.uint8)
                    patch_results.append((x, y, spatial_maps))
            
            del batch_imgs, padded_imgs, outputs
            gc.collect()

    if not patch_results and not target_channels:
        return np.array(wsi_features).mean(axis=0)

    if not patch_results:
        return np.zeros(len(VALID_CHANNELS))

    wsi_mean = np.array(wsi_features).mean(axis=0)
    
    xs = [r[0] for r in patch_results]
    ys = [r[1] for r in patch_results]
    min_x, min_y = min(xs), min(ys)
    
    sample_map = patch_results[0][2][target_channels[0]]
    patch_h, patch_w = sample_map.shape
    
    step_x = get_step(sorted(list(set(xs))), patch_w)
    step_y = get_step(sorted(list(set(ys))), patch_h)
    
    VIS_SCALE = 0.125
    scaled_patch_w = max(1, int(patch_w * VIS_SCALE))
    scaled_patch_h = max(1, int(patch_h * VIS_SCALE))
    
    scale_x = scaled_patch_w / step_x if step_x > 0 else VIS_SCALE
    scale_y = scaled_patch_h / step_y if step_y > 0 else VIS_SCALE
    
    canvas_w = int((max(xs) - min_x) * scale_x) + scaled_patch_w
    canvas_h = int((max(ys) - min_y) * scale_y) + scaled_patch_h
    
    if canvas_w > 15000 or canvas_h > 15000:
        return wsi_mean
    
    for ch in target_channels:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        color_bgr = np.array(COLOR_MAP.get(ch, (255, 255, 255))[::-1], dtype=np.float32) / 255.0
        
        for x, y, smap in patch_results:
            ch_map = smap[ch]
            if VIS_SCALE != 1.0:
                ch_map = cv2.resize(ch_map, (scaled_patch_w, scaled_patch_h), interpolation=cv2.INTER_NEAREST)
            
            colored_patch = (ch_map[..., None] * color_bgr).astype(np.uint8)
            
            start_x = int((x - min_x) * scale_x)
            start_y = int((y - min_y) * scale_y)
            
            end_y = min(start_y + scaled_patch_h, canvas_h)
            end_x = min(start_x + scaled_patch_w, canvas_w)
            
            ph = end_y - start_y
            pw = end_x - start_x
            
            canvas[start_y:end_y, start_x:end_x] = np.maximum(
                canvas[start_y:end_y, start_x:end_x], 
                colored_patch[:ph, :pw]
            )
        
        os.makedirs(os.path.join(output_vis_dir, str(wsi_id)), exist_ok=True)
        cv2.imwrite(os.path.join(output_vis_dir, str(wsi_id), f"{wsi_id}_{ch}.png"), canvas)
        
        del canvas
        gc.collect()
        
    del patch_results
    gc.collect()
            
    return wsi_mean

def main():
    INPUT_DIR = "../01_WSI_Data_Preprocessing/data/patches_normalized"
    OUTPUT_CSV = "./output/WSI_Level_GigaTIME_Features.csv"
    OUTPUT_VIS_DIR = "./output/GigaTIME_WSI_Visualizations"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_gigatime_model(device)
    
    processed_wsis = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            if 'WSI_ID' in df_existing.columns:
                processed_wsis = set(df_existing['WSI_ID'].astype(str).tolist())
        except Exception:
            pass

    wsi_data_map = {}
    if os.path.exists(INPUT_DIR):
        groups = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
        for group in groups:
            group_dir = os.path.join(INPUT_DIR, group)
            wsi_folders = [d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))]
            if group not in wsi_data_map:
                wsi_data_map[group] = {}
            for wsi in wsi_folders:
                wsi_dir = os.path.join(group_dir, wsi)
                patches = glob.glob(os.path.join(wsi_dir, "*.[jp][pn]*[g]")) + glob.glob(os.path.join(wsi_dir, "*.tif"))
                if patches:
                    wsi_data_map[group][wsi] = patches

    all_tasks = []
    for group, wsis in wsi_data_map.items():
        for wsi_id, patch_paths in wsis.items():
            missing_channels = []
            wsi_out_dir = os.path.join(OUTPUT_VIS_DIR, str(wsi_id))
            for ch in VALID_CHANNELS:
                img_path = os.path.join(wsi_out_dir, f"{wsi_id}_{ch}.png")
                if not os.path.exists(img_path):
                    missing_channels.append(ch)
            
            needs_csv = str(wsi_id) not in processed_wsis
            if needs_csv or len(missing_channels) > 0:
                all_tasks.append((group, wsi_id, patch_paths, missing_channels, needs_csv))
    
    all_tasks.sort(key=lambda x: str(x[1]))
    
    if not all_tasks:
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    headers = ['WSI_ID', 'Group', 'Patch_Count'] + VALID_CHANNELS
    if not os.path.exists(OUTPUT_CSV):
        pd.DataFrame(columns=headers).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    for group, wsi_id, patch_paths, missing_channels, needs_csv in tqdm(all_tasks, desc="Total WSIs"):
        try:
            wsi_vector = process_wsi(wsi_id, patch_paths, model, device, missing_channels, OUTPUT_VIS_DIR, batch_size=4)
            
            if needs_csv and wsi_vector is not None:
                row_data = {'WSI_ID': wsi_id, 'Group': group, 'Patch_Count': len(patch_paths)}
                for i, ch_name in enumerate(VALID_CHANNELS):
                    row_data[ch_name] = wsi_vector[i]
                
                pd.DataFrame([row_data]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False, encoding='utf-8-sig')
        except Exception:
            pass

if __name__ == '__main__':
    main()