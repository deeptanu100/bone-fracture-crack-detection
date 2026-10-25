import pandas as pd

excel_path = r"C:\Users\DEEPTANU\.gemini\antigravity-ide\scratch\bone_fracture_detection\BoneFract A Bone Fracture Dataset\BoneFract_Xray_Dataset_Sheets.xlsx"

splits = {}
for split in ['train', 'valid', 'test']:
    df = pd.read_excel(excel_path, sheet_name=f"{split}_labeled_studies", header=None)
    patients = set(df[0].apply(lambda x: x.strip('/').split('/')[3] if len(x.strip('/').split('/')) > 3 else 'Unknown'))
    splits[split] = patients

train_val_overlap = splits['train'].intersection(splits['valid'])
train_test_overlap = splits['train'].intersection(splits['test'])
val_test_overlap = splits['valid'].intersection(splits['test'])

print("Patient ID Overlap Audit:")
print(f"Train & Valid overlap: {len(train_val_overlap)} patients {list(train_val_overlap)[:5]}")
print(f"Train & Test overlap: {len(train_test_overlap)} patients {list(train_test_overlap)[:5]}")
print(f"Valid & Test overlap: {len(val_test_overlap)} patients {list(val_test_overlap)[:5]}")
