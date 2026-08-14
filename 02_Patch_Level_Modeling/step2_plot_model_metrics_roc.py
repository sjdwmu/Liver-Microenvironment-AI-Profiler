import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import cv2
import pandas as pd

warnings.filterwarnings('ignore')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

ROOT_DIR = './data/patch_10_classes'
NUM_CLASSES = 10
OUTPUT_DIR = './output/02_patch_modeling'

WEIGHTS = {
    'ResNet50': os.path.join(OUTPUT_DIR, 'best_resnet50_10class.pth'),
    'EfficientNet-V2-S': os.path.join(OUTPUT_DIR, 'best_efficientnet_v2_s_10class.pth'),
    'Swin-T': os.path.join(OUTPUT_DIR, 'best_swin_t_10class.pth'),
    'ConvNeXt-Tiny': os.path.join(OUTPUT_DIR, 'best_convnext_tiny_10class.pth')
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

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

def cv_imread(file_path):
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception:
        return None

class LiverDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = cv_imread(self.file_paths[idx])
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        if self.transform:
            image = self.transform(image)
        return image, int(self.labels[idx])

def get_tissue_classes():
    if not os.path.isdir(ROOT_DIR):
        raise ValueError(f'Dir not found: {ROOT_DIR}')
    classes = set()
    for entry in os.scandir(ROOT_DIR):
        if entry.is_dir():
            classes.add(entry.name)
    classes = sorted(classes)
    return classes

def load_data():
    classes = get_tissue_classes()
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    all_paths = []
    all_labels = []
    
    for class_name in classes:
        class_dir = os.path.join(ROOT_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for current_dir, _, file_names in os.walk(class_dir):
            for file_name in file_names:
                ext = os.path.splitext(file_name)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    file_path = os.path.join(current_dir, file_name)
                    all_paths.append(file_path)
                    all_labels.append(class_to_idx[class_name])
    return all_paths, all_labels, classes

def get_model(model_name):
    if model_name == 'ResNet50':
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_name == 'EfficientNet-V2-S':
        model = models.efficientnet_v2_s()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif model_name == 'Swin-T':
        model = models.swin_t()
        model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
    elif model_name == 'ConvNeXt-Tiny':
        model = models.convnext_tiny()
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    return model

if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    all_paths, all_labels, classes = load_data()
    
    _, val_paths, _, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = LiverDataset(val_paths, val_labels, transform=val_transform)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=0,
        pin_memory=(str(DEVICE) != 'cpu')
    )

    results_metrics = {}
    results_roc = {}
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_name, weight_file in WEIGHTS.items():
        if not os.path.exists(weight_file):
            continue
            
        model = get_model(model_name).to(DEVICE)
        model.load_state_dict(torch.load(weight_file, map_location=DEVICE, weights_only=True))
        model.eval()
        
        all_preds, all_labels_eval, all_probs = [], [], []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=model_name):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels_eval.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        acc = accuracy_score(all_labels_eval, all_preds)
        prec = precision_score(all_labels_eval, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels_eval, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels_eval, all_preds, average='macro', zero_division=0)
        
        results_metrics[model_name] = [acc, prec, rec, f1]
        
        y_onehot = label_binarize(all_labels_eval, classes=range(NUM_CLASSES))
        all_probs = np.array(all_probs)
        
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)
        valid_classes_count = 0
        
        for i in range(NUM_CLASSES):
            if np.sum(y_onehot[:, i]) > 0:
                fpr_i, tpr_i, _ = roc_curve(y_onehot[:, i], all_probs[:, i])
                mean_tpr += np.interp(fpr_grid, fpr_i, tpr_i)
                valid_classes_count += 1
        
        if valid_classes_count > 0:
            mean_tpr /= valid_classes_count
            mean_tpr[0] = 0.0
            mean_tpr[-1] = 1.0
            macro_auc = auc(fpr_grid, mean_tpr)
        else:
            macro_auc = 0.5
            
        results_roc[model_name] = {'fpr': fpr_grid, 'tpr': mean_tpr, 'auc': macro_auc}

    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("white")

    bar_colors = ['#98D8C8', '#FFFACD', '#D1C4E9', '#F89880']
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    if results_metrics:
        models_list = list(results_metrics.keys())
        
        df_metrics = pd.DataFrame.from_dict(
            results_metrics, 
            orient='index', 
            columns=['Accuracy', 'Precision', 'Recall', 'F1 Score']
        )
        df_metrics.index.name = 'Model_Name'
        df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'Model_Comparison_Metrics.csv'))
        
        x = np.arange(len(models_list))
        
        for i, metric in enumerate(metrics_names):
            fig, ax = plt.subplots(figsize=(8, 6))
            vals = [results_metrics[m][i] for m in models_list]
            
            bars = ax.bar(x, vals, color=bar_colors[:len(models_list)], width=0.6, edgecolor='black', linewidth=0.8)
            
            ax.set_title(f'{metric} Comparison', fontsize=16, pad=5)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(models_list, rotation=15, ha='right', fontsize=11)
            ax.set_ylim(0.0, 1.1)
            
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
                        f'{yval:.3f}', ha='center', va='bottom', fontsize=11)
            
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'Final_{metric.replace(" ", "_")}_Chart.pdf'), format='pdf', bbox_inches='tight', transparent=True)
            plt.savefig(os.path.join(OUTPUT_DIR, f'Final_{metric.replace(" ", "_")}_Chart.svg'), format='svg', bbox_inches='tight', transparent=True)
            plt.close()

    if results_roc:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        sns.set_style("white")
        roc_colors = ['#5DADE2', '#F5B041', '#52BE80', '#E74C3C']
        
        all_tprs = []
        for i, model_name in enumerate(results_roc.keys()):
            data = results_roc[model_name]
            all_tprs.append(data['tpr'])
            plt.plot(data['fpr'], data['tpr'], color=roc_colors[i % len(roc_colors)], 
                     lw=2, label=f"{model_name} (AUC = {data['auc']:.3f})")

        mean_tpr_all = np.mean(all_tprs, axis=0)
        mean_auc_all = auc(fpr_grid, mean_tpr_all)
        
        plt.plot(fpr_grid, mean_tpr_all, color='navy', lw=3.5, 
                 label=f"Average Model (AUC = {mean_auc_all:.3f})")

        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Test Set ROC Curves Comparison', fontsize=16, pad=5)
        plt.legend(loc="lower right", fontsize=12, frameon=True)
        plt.grid(False)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Final_ROC_Comparison.pdf'), format='pdf', bbox_inches='tight', transparent=True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'Final_ROC_Comparison.svg'), format='svg', bbox_inches='tight', transparent=True)
        plt.close()