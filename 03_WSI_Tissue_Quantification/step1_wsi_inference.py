import os
import cv2
import glob
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

CONFIG = {
    'PATHS': {
        'LT-free group': '../01_WSI_Data_Preprocessing/data/patches_normalized/LT-free group',
        'LT-group': '../01_WSI_Data_Preprocessing/data/patches_normalized/LT-group'
    },
    'MODEL_PATH': '../02_Patch_Level_Modeling/output/02_patch_modeling/best_convnext_tiny_10class.pth',
    'OUTPUT_CSV': './output/03_wsi_inference/WSI_Tissue_Composition_English.csv',
    'NUM_CLASSES': 10,
    'BATCH_SIZE': 128,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

CLASSES_ENGLISH = [
    'Necrosis',
    'Lobular and Interlobular Inflammation',
    'Ballooning and Mallory Bodies',
    'Portal Inflammation',
    'Histiocytes',
    'Nodular Regeneration',
    'Cholestasis',
    'Bile Duct Proliferation',
    'Steatosis',
    'Ceroid'
]

def cv_imread(file_path):
    try:
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        if cv_img is not None:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return cv_img
    except:
        return None

class InferenceDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = cv_imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        if self.transform:
            image = self.transform(image)
        return image

def build_model(num_classes):
    model = models.convnext_tiny(weights=None)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE'], weights_only=True))
    return model

def main():
    os.makedirs(os.path.dirname(CONFIG['OUTPUT_CSV']), exist_ok=True)
    
    model = build_model(CONFIG['NUM_CLASSES']).to(CONFIG['DEVICE'])
    model.eval()

    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    results = []

    for group_name, group_dir in CONFIG['PATHS'].items():
        if not os.path.exists(group_dir):
            continue

        wsi_folders = [f.path for f in os.scandir(group_dir) if f.is_dir()]
        
        for wsi_folder in tqdm(wsi_folders, desc=f"Inferencing {group_name}"):
            wsi_id = os.path.basename(wsi_folder)
            patches = glob.glob(os.path.join(wsi_folder, '*.[jp][pn]*[g]'))
            
            if not patches:
                continue

            dataset = InferenceDataset(patches, transform=eval_transform)
            dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)

            class_counts = {i: 0 for i in range(CONFIG['NUM_CLASSES'])}

            with torch.no_grad():
                for inputs in dataloader:
                    inputs = inputs.to(CONFIG['DEVICE'])
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    preds_np = preds.cpu().numpy()
                    unique, counts = np.unique(preds_np, return_counts=True)
                    for u, c in zip(unique, counts):
                        class_counts[u] += c
            
            total_patches = len(patches)
            row = {
                'Group': group_name,
                'WSI_ID': wsi_id,
                'Total_Patches': total_patches
            }
            
            for i in range(CONFIG['NUM_CLASSES']):
                class_name = CLASSES_ENGLISH[i]
                row[f'{class_name}_Count'] = class_counts[i]
                row[class_name] = class_counts[i] / total_patches if total_patches > 0 else 0
                
            results.append(row)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(CONFIG['OUTPUT_CSV'], index=False)

if __name__ == '__main__':
    main()