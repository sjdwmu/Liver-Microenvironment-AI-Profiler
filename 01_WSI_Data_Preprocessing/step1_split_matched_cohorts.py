import os
import re
import shutil
import pandas as pd
from tqdm import tqdm

MATCHED_CSV = './data/clinical_data/Matched data_age+MELD.csv'
MAPPING_EXCEL = './data/clinical_data/肝穿HE片号.xlsx'

DIR_FREE_SRC = './data/raw_wsi/LT-free group'
DIR_LT_SRC = './data/raw_wsi/LT-group'

DIR_FREE_OUT = './data/matched_wsi/LT-free group'
DIR_LT_OUT = './data/matched_wsi/LT-group'

def main():
    os.makedirs(DIR_FREE_OUT, exist_ok=True)
    os.makedirs(DIR_LT_OUT, exist_ok=True)

    df_matched = pd.read_csv(MATCHED_CSV)
    
    free_ids = df_matched[df_matched['treat'].astype(str).str.contains('LT-free', case=False, na=False)]['ID'].astype(str).tolist()
    lt_ids = df_matched[df_matched['treat'].astype(str).str.contains('LT-group', case=False, na=False)]['ID'].astype(str).tolist()

    if not free_ids and not lt_ids:
        free_ids = df_matched[df_matched['treat'].astype(str).str.contains('biopsy|0', case=False, regex=True, na=False)]['ID'].astype(str).tolist()
        lt_ids = df_matched[df_matched['treat'].astype(str).str.contains('transplantation|1', case=False, regex=True, na=False)]['ID'].astype(str).tolist()

    df_mapping = pd.read_excel(MAPPING_EXCEL, header=None)
    mapping_dict = dict(zip(df_mapping[1].astype(str), df_mapping[0].astype(str)))

    free_mrxs_map = {}
    if os.path.exists(DIR_FREE_SRC):
        for root, dirs, files in os.walk(DIR_FREE_SRC):
            for f in files:
                if f.lower().endswith('.mrxs'):
                    base = os.path.splitext(f)[0].strip()
                    free_mrxs_map[base] = os.path.join(root, f)

    for pid in tqdm(free_ids, desc="Processing LT-free group"):
        slide_name = mapping_dict.get(pid)
        if not slide_name or str(slide_name).lower() == 'nan':
            continue
            
        slide_name_str = str(slide_name).strip()
        parts = slide_name_str.split('-')
        suffix = parts[1] if len(parts) == 2 else slide_name_str
        
        src_file = None
        if slide_name_str in free_mrxs_map:
            src_file = free_mrxs_map[slide_name_str]
        elif suffix in free_mrxs_map:
            src_file = free_mrxs_map[suffix]
            
        if src_file:
            src_folder = os.path.splitext(src_file)[0]
            dst_file = os.path.join(DIR_FREE_OUT, os.path.basename(src_file))
            dst_folder = os.path.join(DIR_FREE_OUT, os.path.basename(src_folder))
            
            if os.path.exists(src_folder) and not os.path.exists(dst_folder):
                shutil.copytree(src_folder, dst_folder)
            
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)

    if os.path.exists(DIR_LT_SRC):
        all_lt_items = os.listdir(DIR_LT_SRC)
    else:
        all_lt_items = []

    for pid in tqdm(lt_ids, desc="Processing LT-group"):
        pattern = re.compile(rf"^{pid}( \(\d+\))?(\.mrxs)?$")
        
        matched_items = [item for item in all_lt_items if pattern.match(item)]
        if matched_items:
            for item in matched_items:
                src_path = os.path.join(DIR_LT_SRC, item)
                dst_path = os.path.join(DIR_LT_OUT, item)
                
                if os.path.isdir(src_path) and not os.path.exists(dst_path):
                    shutil.copytree(src_path, dst_path)
                elif os.path.isfile(src_path) and not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)

if __name__ == '__main__':
    main()