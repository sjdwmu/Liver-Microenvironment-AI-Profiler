import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
import warnings

warnings.filterwarnings('ignore')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

CSV_PATH = './output/03_wsi_inference/WSI_Tissue_Composition_English.csv'
OUTPUT_DIR = './output/statistical_analysis'

CLASSES = [
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

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(CSV_PATH):
        return

    df = pd.read_csv(CSV_PATH)

    df['Group_Name'] = df['Group']

    df['Label'] = df['Group_Name'].map({'LT-free group': 0, 'LT-group': 1})

    valid_cols = [c for c in CLASSES if c in df.columns]
    if not valid_cols:
        return

    X = df[valid_cols]
    y = df['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, penalty='l2', C=1.0)
    lr.fit(X_scaled, y)

    coefs = lr.coef_[0]
    importances = np.abs(coefs)

    df_results = pd.DataFrame({
        'Feature': valid_cols,
        'Raw_Coefficient': coefs,
        'Importance_Score': importances
    }).sort_values(by='Importance_Score', ascending=False)

    df_results.to_csv(os.path.join(OUTPUT_DIR, 'Full_Feature_Importances_Raw_Data.csv'), index=False)

    sns.set_style("white")
    plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance_Score', y='Feature', data=df_results, color='#89ABCD', ax=ax1, edgecolor='black', linewidth=1)
    ax1.set_title('Microenvironment Feature Importance of the Full Cohort', fontsize=16, pad=20)
    ax1.set_xlabel('Importance Score', fontsize=14)
    ax1.set_ylabel('')
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Full_Feature_Importances_Absolute_Style.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Full_Feature_Importances_Absolute_Style.pdf'), format='pdf', bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Full_Feature_Importances_Absolute_Style.svg'), format='svg', bbox_inches='tight', transparent=True)
    plt.close()

    df_raw = df_results.sort_values(by='Raw_Coefficient', ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    colors = ['#E94E77' if c > 0 else '#4A90E2' for c in df_raw['Raw_Coefficient']]
    sns.barplot(x='Raw_Coefficient', y='Feature', data=df_raw, palette=colors, ax=ax2, edgecolor='black', linewidth=1)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    ax2.set_title('Microenvironment Feature Weights of the Full Cohort', fontsize=16, pad=20)
    ax2.set_xlabel('Raw Coefficient (Weight)', fontsize=14)
    ax2.set_ylabel('')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('black')
    ax2.spines['bottom'].set_color('black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Full_Feature_Weights_Directional.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Full_Feature_Weights_Directional.pdf'), format='pdf', bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Full_Feature_Weights_Directional.svg'), format='svg', bbox_inches='tight', transparent=True)
    plt.close()

    df_melted = pd.melt(df, id_vars=['Group_Name'], value_vars=valid_cols,
                        var_name='Feature', value_name='Ratio')

    fig3, ax3 = plt.subplots(figsize=(16, 8))
    
    palette = {'LT-free group': '#BFE8FC', 'LT-group': '#FFC0B2'}

    sns.boxplot(x='Feature', y='Ratio', hue='Group_Name', data=df_melted,
                palette=palette, width=0.5, fliersize=0, linewidth=1.2,
                ax=ax3, order=CLASSES, hue_order=['LT-free group', 'LT-group'],
                boxprops={'edgecolor': 'black', 'alpha': 0.9})

    global_max = df_melted['Ratio'].max()
    ax3.set_ylim(-0.02, global_max * 1.35)

    new_xticklabels = []
    for i, feature_name in enumerate(CLASSES):
        if feature_name not in valid_cols:
            new_xticklabels.append(feature_name)
            continue

        group1_vals = df[df['Group_Name'] == 'LT-free group'][feature_name].values
        group2_vals = df[df['Group_Name'] == 'LT-group'][feature_name].values

        if len(group1_vals) == 0 or len(group2_vals) == 0:
            p = 1.0
        else:
            _, p = mannwhitneyu(group1_vals, group2_vals, alternative='two-sided')

        if p < 0.001:
            sig = "***"
            p_text = "p < 0.001"
        elif p < 0.01:
            sig = "**"
            p_text = f"p = {p:.3f}"
        elif p < 0.05:
            sig = "*"
            p_text = f"p = {p:.3f}"
        else:
            sig = "ns"
            p_text = f"p = {p:.3f}"

        y_max = df_melted[df_melted['Feature'] == feature_name]['Ratio'].max()
        if pd.isna(y_max):
            y_max = 0
        y_pos = y_max + (global_max * 0.05) if y_max > 0 else 0.05

        plt.plot([i - 0.2, i + 0.2], [y_pos, y_pos], color='black', lw=1.5)
        plt.text(i, y_pos + (global_max * 0.015),
                 sig, ha='center', va='bottom', color='black', fontsize=16)

        new_xticklabels.append(f"{feature_name}\n({p_text})")

    ax3.set_xticks(range(len(CLASSES)))
    ax3.set_xticklabels(new_xticklabels, fontsize=10, rotation=25, ha='right')

    ax3.set_title('Quantitative Comparison of Pathology Features of the Full Cohort', fontsize=16, pad=20)
    ax3.set_ylabel('Tissue Ratio', fontsize=14)
    ax3.set_xlabel('')

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('black')
    ax3.spines['bottom'].set_color('black')

    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles=handles, labels=labels, title=None,
              loc='upper right', frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Final_Boxplots_All10_Beautified.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Final_Boxplots_All10_Beautified.pdf'), format='pdf', bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Final_Boxplots_All10_Beautified.svg'), format='svg', bbox_inches='tight', transparent=True)
    plt.close()

if __name__ == '__main__':
    main()