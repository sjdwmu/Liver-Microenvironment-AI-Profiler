import os
import pandas as pd

CSV_PATH = r"E:\sjd\match\WSI_Tissue_Composition.csv"
OUTPUT_PATH = r"E:\sjd\match\WSI_Tissue_Composition_English.csv"


if __name__ == '__main__':
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        
        if 'Group' in df.columns:
            df['Group'] = df['Group'].replace({'Survival': 'Liver biopsy', 'Death': 'Liver transplantation'})
            
        classes = [
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
        
        rename_dict = {}
        for i, name in enumerate(classes):
            rename_dict[f'Class_{i}_Ratio'] = name
            rename_dict[f'Class_{i}_Count'] = f'{name}_Count'
            
        df = df.rename(columns=rename_dict)
        df.to_csv(OUTPUT_PATH, index=False)