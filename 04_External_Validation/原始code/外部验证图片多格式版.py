import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings('ignore')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

TRAIN_CSV = r"E:\sjd\match\WSI_Tissue_Composition.csv"
VAL_CSV = r"E:\sjd\match\Val_WSI_Tissue_Composition.csv"
OUTPUT_DIR = r"E:\sjd\match"

def map_labels(df):
    group_map = {
        'Liver biopsy': 0, 
        'Liver transplantation': 1,
        'Survival': 0,
        'Death': 1,
        'Biopsy_Survival': 0,
        'Resected_Survival': 1
    }
    df['Label'] = df['Group'].map(group_map)
    return df.dropna(subset=['Label'])

def main():
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        return

    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)

    df_train = map_labels(df_train)
    df_val = map_labels(df_val)

    feature_cols = [f'Class_{i}_Ratio' for i in range(10)]

    X_train = df_train[feature_cols]
    y_train = df_train['Label']
    
    X_val = df_val[feature_cols]
    y_val = df_val['Label']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, penalty='l2', C=1.0)
    lr.fit(X_train_scaled, y_train)

    y_val_prob = lr.predict_proba(X_val_scaled)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
    val_auc = auc(fpr, tpr)
    
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    
    y_val_pred_optimal = (y_val_prob >= optimal_threshold).astype(int)
    
    acc_optimal = accuracy_score(y_val, y_val_pred_optimal)
    
    df_roc_raw = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Threshold': thresholds
    })
    df_roc_raw.to_csv(os.path.join(OUTPUT_DIR, 'External_Validation_ROC_Raw_Data.csv'), index=False)

    plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("white")

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='#E94E77', lw=3, label=f'AUC = {val_auc:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='grey', alpha=0.5)
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], color='blue', s=100, label=f'Optimal Cutoff ({optimal_threshold:.2f})', zorder=5)
    plt.title('External Validation ROC Curve (Logistic Regression)', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right")
    sns.despine()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'External_Validation_ROC_LR.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'External_Validation_ROC_LR.pdf'), format='pdf', bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'External_Validation_ROC_LR.svg'), format='svg', bbox_inches='tight', transparent=True)
    plt.close()

    cm = confusion_matrix(y_val, y_val_pred_optimal)
    
    df_cm_raw = pd.DataFrame(cm, index=['Actual_Survival', 'Actual_Death'], columns=['Predicted_Survival', 'Predicted_Death'])
    df_cm_raw.to_csv(os.path.join(OUTPUT_DIR, 'External_Validation_CM_Raw_Data.csv'))

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Survival', 'Death'])
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f'External Validation CM (Threshold: {optimal_threshold:.2f})', pad=20)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'External_Validation_CM_LR.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'External_Validation_CM_LR.pdf'), format='pdf', bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'External_Validation_CM_LR.svg'), format='svg', bbox_inches='tight', transparent=True)
    plt.close()

if __name__ == '__main__':
    main()