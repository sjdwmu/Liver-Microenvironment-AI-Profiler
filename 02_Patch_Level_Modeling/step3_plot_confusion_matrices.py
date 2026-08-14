import os
import cv2
import glob
import torch
import numpy as np
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    'ROOT_DIRS': [
        './data/patch_10_classes'
    ],
    'BATCH_SIZE': 64,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'MODEL_PATH': "./output/02_patch_modeling/best_convnext_tiny_10class.pth",
    'OUTPUT_DIR': "./output/02_patch_modeling"
}

CLASS_TRANSLATION = {
    '坏死': 'Necrosis',
    '小叶内、小叶间炎症': 'Lobular/Interlobular Inflammation',
    '气球样变性和mallory小体': 'Ballooning & Mallory Bodies',
    '汇管区炎症': 'Portal Inflammation',
    '组织细胞': 'Histiocytes',
    '结节状再生': 'Nodular Regeneration',
    '胆汁淤积': 'Cholestasis',
    '胆管增生': 'Bile Duct Proliferation',
    '脂肪变性': 'Steatosis',
    '蜡质': 'Ceroid'
}

def cv_imread(file_path):
    try:
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        if cv_img is not None:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return cv_img
    except:
        return None

def get_classes(dirs):
    classes = set()
    for d in dirs:
        if os.path.exists(d):
            for folder in os.listdir(d):
                if os.path.isdir(os.path.join(d, folder)):
                    classes.add(folder)
    return sorted(list(classes))

class LiverDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
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
        label = self.labels[idx]
        return image, label

def load_data():
    classes = get_classes(CONFIG['ROOT_DIRS'])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    all_paths = []
    all_labels = []
    for root_dir in CONFIG['ROOT_DIRS']:
        if not os.path.exists(root_dir):
            continue
        for cls_name in classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.exists(cls_dir):
                files = glob.glob(os.path.join(cls_dir, '*.[jp][pn]*[g]'))
                all_paths.extend(files)
                all_labels.extend([class_to_idx[cls_name]] * len(files))
    return all_paths, all_labels, classes

def build_model(num_classes):
    model = models.convnext_tiny(weights=None)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE'], weights_only=True))
    return model

def main():
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    all_paths, all_labels, classes = load_data()
    num_classes = len(classes)
    
    if len(all_paths) == 0:
        return

    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = LiverDataset(all_paths, all_labels, transform=eval_transform)
    dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)
    
    model = build_model(num_classes).to(CONFIG['DEVICE'])
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(CONFIG['DEVICE'])
            labels = labels.to(CONFIG['DEVICE'])
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    
    english_classes = [CLASS_TRANSLATION.get(cls, cls) for cls in classes]
    
    cm = confusion_matrix(all_targets, all_preds)
    
    cm_df = pd.DataFrame(cm, index=english_classes, columns=english_classes)
    csv_save_path = os.path.join(CONFIG['OUTPUT_DIR'], 'convnext_tiny_confusion_matrix_raw_data.csv')
    cm_df.to_csv(csv_save_path)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=english_classes)
    
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
    
    plt.title("ConvNeXt-Tiny 10-Class Confusion Matrix", fontsize=16, pad=20)
    plt.tight_layout()
    
    cm_save_path_png = os.path.join(CONFIG['OUTPUT_DIR'], 'convnext_tiny_confusion_matrix_english.png')
    plt.savefig(cm_save_path_png, dpi=300, bbox_inches='tight')
    
    cm_save_path_pdf = os.path.join(CONFIG['OUTPUT_DIR'], 'convnext_tiny_confusion_matrix_english.pdf')
    plt.savefig(cm_save_path_pdf, format='pdf', bbox_inches='tight', transparent=True)
    
    cm_save_path_svg = os.path.join(CONFIG['OUTPUT_DIR'], 'convnext_tiny_confusion_matrix_english.svg')
    plt.savefig(cm_save_path_svg, format='svg', bbox_inches='tight', transparent=True)
    
    plt.close()
    
if __name__ == '__main__':
    main()