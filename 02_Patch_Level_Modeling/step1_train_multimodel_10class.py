import os
import cv2
import ssl
import json
import torch
import certifi
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

def _create_certifi_ssl_context(*args, **kwargs):
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = _create_certifi_ssl_context

CONFIG = {
    'ROOT_DIR': './data/patch_10_classes',
    'BATCH_SIZE': 32,
    'EPOCHS': 50,
    'LR': 0.0002,
    'WEIGHT_DECAY': 1e-4,
    'PATIENCE': 12,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'MODELS_TO_TRAIN': [
        'efficientnet_v2_s',
        'swin_t',
        'resnet50',
        'convnext_tiny'
    ]
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

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'
}

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

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

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
        return (image, int(self.labels[idx]))

def get_tissue_classes():
    root_dir = CONFIG['ROOT_DIR']
    if not os.path.isdir(root_dir):
        raise ValueError(f'训练集根目录不存在: {root_dir}')

    classes = set()
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            classes.add(entry.name)

    classes = sorted(classes)
    if len(classes) != 10:
        raise ValueError(f'检测到 {len(classes)} 个组织类型，而不是预期的 10 个。检测结果: {classes}')

    untranslated = [class_name for class_name in classes if class_name not in CLASS_TRANSLATION]
    if untranslated:
        raise ValueError(f'以下组织类型没有英文翻译: {untranslated}')

    return classes

def load_data():
    classes = get_tissue_classes()
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    all_paths = []
    all_labels = []

    counts = {class_name: 0 for class_name in classes}

    for class_name in classes:
        class_dir = os.path.join(CONFIG['ROOT_DIR'], class_name)
        if not os.path.isdir(class_dir):
            continue
        for (current_dir, _, file_names) in os.walk(class_dir):
            for file_name in file_names:
                ext = os.path.splitext(file_name)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    file_path = os.path.join(current_dir, file_name)
                    all_paths.append(file_path)
                    all_labels.append(class_to_idx[class_name])
                    counts[class_name] += 1

    if len(all_paths) == 0:
        raise ValueError('没有扫描到任何图片。')

    total_counts = np.bincount(np.asarray(all_labels, dtype=np.int64), minlength=len(classes))
    empty_classes = np.where(total_counts == 0)[0]

    if len(empty_classes) > 0:
        empty_names = [classes[i] for i in empty_classes]
        raise ValueError(f'以下组织类型没有扫描到图片: {empty_names}')

    print(f'训练集根目录: {CONFIG["ROOT_DIR"]}')
    print(f'检测到组织类型数量: {len(classes)}')
    
    print('各组织类型图片数量:')
    for class_name in classes:
        print(f'  {CLASS_TRANSLATION[class_name]}: {counts[class_name]}')

    print(f'总图片数量: {len(all_paths)}')

    return (all_paths, all_labels, classes)

def build_model(model_name, num_classes):
    if model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'swin_t':
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f'不支持的模型: {model_name}')
    return model

def plot_curves(history, model_name, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'{model_name.upper()} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title(f'{model_name.upper()} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_auc'], label='Val AUC')
    plt.title(f'{model_name.upper()} - AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_learning_curves.png'), dpi=150)
    plt.close()

def save_class_mapping(classes, save_dir):
    mapping = {
        str(i): {'english': CLASS_TRANSLATION[class_name]}
        for i, class_name in enumerate(classes)
    }
    with open(os.path.join(save_dir, 'class_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

def main():
    print(f'Device: {CONFIG["DEVICE"]}')
    print(f'SSL CA file: {certifi.where()}')
    
    all_paths, all_labels, classes = load_data()
    num_classes = len(classes)
    
    all_labels_np = np.asarray(all_labels, dtype=np.int64)
    class_counts = np.bincount(all_labels_np, minlength=num_classes)

    if np.any(class_counts < 2):
        bad = [CLASS_TRANSLATION[classes[i]] for i in np.where(class_counts < 2)[0]]
        raise ValueError(f'以下组织类型图片数量不足以划分训练集和验证集: {bad}')

    class_weights = len(all_labels_np) / (num_classes * class_counts.astype(np.float64))

    print('类别权重:')
    for i, class_name in enumerate(classes):
        print(f'  {CLASS_TRANSLATION[class_name]}: {class_weights[i]:.6f}')

    class_weights_tensor = torch.tensor(
        class_weights, dtype=torch.float32, device=CONFIG['DEVICE']
    )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    print(f'训练集图片数量: {len(train_paths)}')
    print(f'验证集图片数量: {len(val_paths)}')

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = LiverDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = LiverDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=0,
        pin_memory=(CONFIG['DEVICE'] == 'cuda')
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=0,
        pin_memory=(CONFIG['DEVICE'] == 'cuda')
    )

    save_dir = './output/02_patch_modeling'
    os.makedirs(save_dir, exist_ok=True)
    save_class_mapping(classes, save_dir)

    for model_name in CONFIG['MODELS_TO_TRAIN']:
        print(f'\n开始训练模型: {model_name}')
        try:
            model = build_model(model_name, num_classes).to(CONFIG['DEVICE'])
        except Exception as e:
            raise RuntimeError(f'{model_name} 预训练权重加载失败: {e}') from e

        save_path = os.path.join(save_dir, f'best_{model_name}_10class.pth')
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=CONFIG['LR'], 
            weight_decay=CONFIG['WEIGHT_DECAY']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        early_stopping = EarlyStopping(patience=CONFIG['PATIENCE'])

        history = {
            'train_loss': [], 'val_loss': [], 
            'train_acc': [], 'val_acc': [], 'val_auc': []
        }

        for epoch in range(CONFIG['EPOCHS']):
            model.train()
            train_loss = 0.0
            train_correct = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(CONFIG['DEVICE'], non_blocking=True)
                labels = labels.to(CONFIG['DEVICE'], non_blocking=True)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()

            epoch_t_loss = train_loss / len(train_dataset)
            epoch_t_acc = train_correct / len(train_dataset)

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_labels_all = []
            val_probs_all = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(CONFIG['DEVICE'], non_blocking=True)
                    labels = labels.to(CONFIG['DEVICE'], non_blocking=True)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    probs = torch.softmax(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()

                    val_labels_all.extend(labels.cpu().numpy().tolist())
                    val_probs_all.extend(probs.cpu().numpy().tolist())

            epoch_v_loss = val_loss / len(val_dataset)
            epoch_v_acc = val_correct / len(val_dataset)

            try:
                val_labels_np = np.asarray(val_labels_all, dtype=np.int64)
                val_probs_np = np.asarray(val_probs_all, dtype=np.float64)
                epoch_v_auc = roc_auc_score(
                    val_labels_np,
                    val_probs_np,
                    multi_class='ovr',
                    labels=np.arange(num_classes)
                )
            except Exception:
                epoch_v_auc = 0.0

            history['train_loss'].append(epoch_t_loss)
            history['train_acc'].append(epoch_t_acc)
            history['val_loss'].append(epoch_v_loss)
            history['val_acc'].append(epoch_v_acc)
            history['val_auc'].append(epoch_v_auc)

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch {epoch + 1}/{CONFIG["EPOCHS"]} '
                f'Train Loss: {epoch_t_loss:.4f} '
                f'Train Acc: {epoch_t_acc:.4f} '
                f'Val Loss: {epoch_v_loss:.4f} '
                f'Val Acc: {epoch_v_acc:.4f} '
                f'Val AUC: {epoch_v_auc:.4f} '
                f'LR: {current_lr:.8f}'
            )

            scheduler.step(epoch_v_loss)
            early_stopping(epoch_v_loss, model, save_path)

            if early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        plot_curves(history, model_name, save_dir)
        print(f'模型训练完成: {model_name}')
        print(f'最佳模型保存位置: {save_path}')

    print('\n全部模型训练完成。')

if __name__ == '__main__':
    main()