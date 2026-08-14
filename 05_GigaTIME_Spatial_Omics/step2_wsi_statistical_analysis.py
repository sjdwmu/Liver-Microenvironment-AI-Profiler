import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings('ignore')

def main():
    CSV_PATH = "./output/WSI_Level_GigaTIME_Features.csv"
    OUTPUT_DIR = "./output/statistical_analysis"
    
    if not os.path.exists(CSV_PATH):
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    
    VALID_CHANNELS = [
        'CD4', 'CD8', 'CD20', 'T-bet', 'Caspase3-D', 'CK', 
        'PD-1', 'PD-L1', 'Ki67', 'CD3', 'CD138', 'Tryptase'
    ]
    
    if 'Group' not in df.columns:
        return
    
    sns.set_style("white")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    palette = {'LT-free group': '#BFE8FC', 'LT-group': '#FFC0B2'}
    
    df_melted = pd.melt(df, id_vars=['Group'], value_vars=VALID_CHANNELS, 
                        var_name='Marker', value_name='Expression')
    df_melted['Expression'] = pd.to_numeric(df_melted['Expression'], errors='coerce').fillna(0)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    sns.boxplot(x='Marker', y='Expression', hue='Group', data=df_melted,
                palette=palette, width=0.6, fliersize=0, linewidth=1.2, 
                ax=ax, order=VALID_CHANNELS, hue_order=['LT-free group', 'LT-group'])
                
    global_max = df_melted['Expression'].max()
    if pd.isna(global_max): global_max = 0
    ax.set_ylim(-0.02, global_max * 1.35)
    
    new_xticklabels = []
    for i, marker in enumerate(VALID_CHANNELS):
        g1_vals = df[df['Group'] == 'LT-free group'][marker].dropna().values
        g2_vals = df[df['Group'] == 'LT-group'][marker].dropna().values
        
        if len(g1_vals) < 3 or len(g2_vals) < 3:
            p = 1.0
        else:
            _, p = mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
        
        if p < 0.001:
            sig = "***"
            p_text = "p<0.001"
        elif p < 0.01:
            sig = "**"
            p_text = f"p={p:.3f}"
        elif p < 0.05:
            sig = "*"
            p_text = f"p={p:.3f}"
        else:
            sig = "ns"
            p_text = f"p={p:.3f}"
            
        y_max = df_melted[df_melted['Marker'] == marker]['Expression'].max()
        if pd.isna(y_max): y_max = 0
        y_pos = y_max + (global_max * 0.05) if y_max > 0 else 0.05
        
        ax.plot([i - 0.2, i + 0.2], [y_pos, y_pos], color='black', lw=1.5)
        ax.text(i, y_pos + (global_max * 0.015), 
                 sig, ha='center', va='bottom', color='black', fontsize=12)
        
        new_xticklabels.append(f"{marker}\n({p_text})")
        
    ax.set_xticks(range(len(VALID_CHANNELS)))
    ax.set_xticklabels(new_xticklabels, fontsize=10, rotation=45, ha='right', rotation_mode="anchor")
    
    ax.set_title('GigaTIME Molecular Expressions: LT-free group vs LT-group', fontsize=16, pad=20)
    ax.set_ylabel('Expression Level', fontsize=12)
    ax.set_xlabel('')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title=None, 
              loc='upper right', frameon=False, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Expression_Boxplot.png'), dpi=300)
    plt.close()
    
    results = []
    group_a = df[df['Group'] == 'LT-free group']
    group_c = df[df['Group'] == 'LT-group']
    
    for ch in VALID_CHANNELS:
        v_a = group_a[ch].dropna()
        v_c = group_c[ch].dropna()
        
        if len(v_a) > 2 and len(v_c) > 2:
            _, p_val = mannwhitneyu(v_a, v_c, alternative='two-sided')
            log_p = -np.log10(p_val) if (p_val > 0 and not np.isnan(p_val)) else 0
        else:
            log_p = 0
            
        results.append({'Marker': ch, 'LogP': log_p})
        
    res_df = pd.DataFrame(results).sort_values(by='LogP', ascending=True)
    
    temp_df = df[['Group'] + VALID_CHANNELS].dropna()
    if not temp_df.empty:
        X = temp_df[VALID_CHANNELS]
        y = (temp_df['Group'] == 'LT-group').astype(int).values
        
        log_p_multi = 0
        if len(np.unique(y)) == 2 and len(y) > 2:
            try:
                X_scaled = (X - X.mean()) / (X.std() + 1e-9)
                prior = y.mean()
                null_prob = np.full(len(y), prior)
                llf_null = -log_loss(y, null_prob, normalize=False)
                
                clf = LogisticRegression(penalty='l2', C=1e5, solver='lbfgs', max_iter=2000)
                clf.fit(X_scaled, y)
                full_prob = clf.predict_proba(X_scaled)
                llf_full = -log_loss(y, full_prob, normalize=False)
                
                llr = -2 * (llf_null - llf_full)
                p_val_multi = chi2.sf(llr, df=X_scaled.shape[1])
                
                if p_val_multi > 0 and not np.isnan(p_val_multi):
                    log_p_multi = -np.log10(p_val_multi)
                elif p_val_multi == 0:
                    log_p_multi = 50 
            except Exception:
                log_p_multi = 0

        multi_row = pd.DataFrame([{'Marker': 'GigaTIME\nsignature', 'LogP': log_p_multi}])
        res_df = pd.concat([res_df, multi_row], ignore_index=True)
        
    fig, ax = plt.subplots(figsize=(5, 8))
    
    bars = ax.barh(res_df['Marker'], res_df['LogP'], color='#89ABCD', edgecolor='black', linewidth=1)
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('-log10(p-value)', fontsize=12)
    
    ax.set_yticks(np.arange(len(res_df)))
    ax.set_yticklabels(res_df['Marker'], rotation=0, ha='right', va='center', fontsize=10)
    
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    plt.title('Feature Importance', fontsize=14, fontweight='bold', pad=40)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Feature_Importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    res_df.sort_values(by='LogP', ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, 'Feature_Importance_P_Values.csv'), 
        index=False
    )

if __name__ == '__main__':
    main()