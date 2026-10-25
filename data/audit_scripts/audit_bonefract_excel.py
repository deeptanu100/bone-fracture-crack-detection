import pandas as pd

excel_path = r"C:\Users\DEEPTANU\.gemini\antigravity-ide\scratch\bone_fracture_detection\BoneFract A Bone Fracture Dataset\BoneFract_Xray_Dataset_Sheets.xlsx"

xl = pd.ExcelFile(excel_path)
print("Sheet names:", xl.sheet_names)

for sheet in xl.sheet_names:
    df = xl.parse(sheet)
    print(f"\n--- Sheet: {sheet} ---")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head(3))
