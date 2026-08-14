import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings('ignore')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

CSV_PATH = '../03_WSI_Tissue_Quantification/output/03_wsi_inference/WSI_Tissue_Composition_English.csv'
OUTPUT_DIR = './output'

CLASSES_EN = [
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

def map_labels(df):
    group_map = {
        'LT-free group': 0, 
        'LT-group': 1
    }
    df['Label'] = df['Group'].map(group_map)
    return df.dropna(subset=['Label'])

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(CSV_PATH):
        return

    df = map_labels(pd.read_csv(CSV_PATH))

    feature_cols = [c for c in CLASSES_EN if c in df.columns]
    
    if not feature_cols:
        return

    X = df[feature_cols]
    y = df['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, penalty='l2', C=1.0)
    lr.fit(X_scaled, y)

    explainer = shap.LinearExplainer(lr, X_scaled)
    shap_values = explainer.shap_values(X_scaled)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    coefs = lr.coef_[0]

    features_info = []
    for i in range(len(feature_cols)):
        features_info.append({
            'Feature': feature_cols[i],
            'Importance': mean_abs_shap[i],
            'Direction': 'Risk' if coefs[i] > 0 else 'Protective'
        })
        
    df_plot = pd.DataFrame(features_info).sort_values(by='Importance', ascending=True)

    plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    color_map = {'Risk': '#E94E77', 'Protective': '#4A90E2'}
    colors = [color_map[d] for d in df_plot['Direction']]
    
    bars = ax.barh(df_plot['Feature'], df_plot['Importance'], color=colors, height=0.6)
    
    ax.set_xlabel('Mean |SHAP| value', fontsize=13)
    ax.set_title('Microenvironment SHAP Feature Contribution (Full Cohort)', fontsize=16, pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    risk_patch = mpatches.Patch(color='#E94E77', label='Positive Factor (Promotes LT-group)')
    prot_patch = mpatches.Patch(color='#4A90E2', label='Negative Factor (Promotes LT-free group)')
    ax.legend(handles=[risk_patch, prot_patch], loc='lower right', fontsize=12)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                ha='left', va='center', fontsize=11, color='black')

    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'SHAP_Feature_Importance_Bar_EN.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'SHAP_Feature_Importance_Bar_EN.pdf'), format='pdf', bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'SHAP_Feature_Importance_Bar_EN.svg'), format='svg', bbox_inches='tight', transparent=True)
    plt.close()

if __name__ == '__main__':
    main()