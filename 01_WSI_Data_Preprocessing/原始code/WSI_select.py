import os
import re
import shutil
import pandas as pd
from tqdm import tqdm

MATCHED_CSV = r'D:\sjd\project\liver_failure\match\Matched data_age+MELD.csv'
MAPPING_EXCEL = r'D:\sjd\project\liver_failure\2.0 data\肝穿刺\肝穿HE片号.xlsx'

DIR_BIOPSY_SRC = r'D:\sjd\project\liver_failure\2.0 data\肝穿刺\HE'
DIR_TRANS_SRC = r'D:\sjd\project\liver_failure\2.0 data\code\code3.0\36vs81\Selected_WSIs_All_Death'

DIR_BIOPSY_OUT = r'D:\sjd\project\liver_failure\match\Liver biopsy'
DIR_TRANS_OUT = r'D:\sjd\project\liver_failure\match\Liver transplantation'

def main():
    os.makedirs(DIR_BIOPSY_OUT, exist_ok=True)
    os.makedirs(DIR_TRANS_OUT, exist_ok=True)

    df_matched = pd.read_csv(MATCHED_CSV)
    
    biopsy_ids = df_matched[df_matched['treat'].astype(str).str.contains('biopsy', case=False, na=False)]['ID'].astype(str).tolist()
    trans_ids = df_matched[df_matched['treat'].astype(str).str.contains('transplantation', case=False, na=False)]['ID'].astype(str).tolist()

    print(f"Found Biopsy IDs: {len(biopsy_ids)}")
    print(f"Found Transplantation IDs: {len(trans_ids)}")

    df_mapping = pd.read_excel(MAPPING_EXCEL, header=None)
    mapping_dict = dict(zip(df_mapping[1].astype(str), df_mapping[0].astype(str)))

    for pid in tqdm(biopsy_ids, desc="Processing Liver biopsy"):
        slide_name = mapping_dict.get(pid)
        if not slide_name or str(slide_name).lower() == 'nan':
            continue
            
        parts = str(slide_name).split('-')
        if len(parts) != 2:
            continue
            
        prefix, suffix = parts[0], parts[1]
        src_folder = os.path.join(DIR_BIOPSY_SRC, prefix, suffix)
        src_file = os.path.join(DIR_BIOPSY_SRC, prefix, f"{suffix}.mrxs")
        
        target_dir = os.path.join(DIR_BIOPSY_OUT, pid)
        os.makedirs(target_dir, exist_ok=True)
        
        if os.path.exists(src_folder):
            dst_folder = os.path.join(target_dir, suffix)
            if not os.path.exists(dst_folder):
                shutil.copytree(src_folder, dst_folder)
        
        if os.path.exists(src_file):
            dst_file = os.path.join(target_dir, f"{suffix}.mrxs")
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)

    if os.path.exists(DIR_TRANS_SRC):
        all_trans_items = os.listdir(DIR_TRANS_SRC)
    else:
        all_trans_items = []

    for pid in tqdm(trans_ids, desc="Processing Liver transplantation"):
        target_dir = os.path.join(DIR_TRANS_OUT, pid)
        pattern = re.compile(rf"^{pid}( \(\d+\))?(\.mrxs)?$")
        
        matched_items = [item for item in all_trans_items if pattern.match(item)]
        if matched_items:
            os.makedirs(target_dir, exist_ok=True)
            for item in matched_items:
                src_path = os.path.join(DIR_TRANS_SRC, item)
                dst_path = os.path.join(target_dir, item)
                
                if os.path.isdir(src_path) and not os.path.exists(dst_path):
                    shutil.copytree(src_path, dst_path)
                elif os.path.isfile(src_path) and not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)

if __name__ == '__main__':
    main()