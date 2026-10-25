import pandas as pd
import re

excel_path = r"C:\Users\DEEPTANU\.gemini\antigravity-ide\scratch\bone_fracture_detection\BoneFract A Bone Fracture Dataset\BoneFract_Xray_Dataset_Sheets.xlsx"

for split in ['train', 'valid', 'test']:
    df = pd.read_excel(excel_path, sheet_name=f"{split}_labeled_studies", header=None)
    df.columns = ['study_path', 'label']
    
    # parse body part and patient
    def parse_info(p):
        parts = p.strip('/').split('/')
        # parts: ['BoneFract A Bone Fracture Dataset', split, body_part, patient_id, label]
        body_part = parts[2] if len(parts) > 2 else 'Unknown'
        patient = parts[3] if len(parts) > 3 else 'Unknown'
        return body_part, patient
    
    parsed = df['study_path'].apply(parse_info)
    df['body_part'] = [x[0] for x in parsed]
    df['patient_id'] = [x[1] for x in parsed]
    
    print(f"\n=================== {split.upper()} SET ===================")
    print(f"Total samples: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print("Class Balance:")
    print(df['label'].value_counts(normalize=True).round(4).to_dict())
    print(df['label'].value_counts().to_dict())
    print("Body Part Breakdown:")
    print(df['body_part'].value_counts().to_dict())
    print("Class Balance by Body Part (Fracture Rate %):")
    grouped = df.groupby('body_part')['label'].agg(['count', 'mean']).rename(columns={'mean': 'fracture_rate'})
    grouped['fracture_rate'] = (grouped['fracture_rate'] * 100).round(2)
    print(grouped)
