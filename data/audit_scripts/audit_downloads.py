import os
import glob
import zipfile

bf_dir = r"C:\Users\DEEPTANU\Downloads\BoneFract A Bone Fracture Dataset"
c17_dir = r"C:\Users\DEEPTANU\Downloads\Human Bone Fracture C17 Dataset"

print("=== BoneFract Directory ===")
for root, dirs, files in os.walk(bf_dir):
    print(f"Path: {root}")
    print(f"  Dirs ({len(dirs)}): {dirs[:10]}")
    print(f"  Files ({len(files)}): {files[:10]}")
    for f in files:
        if f.endswith('.zip'):
            full_p = os.path.join(root, f)
            sz_gb = os.path.getsize(full_p) / (1024**3)
            print(f"  ZIP file found: {f} ({sz_gb:.2f} GB)")
            with zipfile.ZipFile(full_p, 'r') as zf:
                namelist = zf.namelist()
                print(f"  Total items inside zip: {len(namelist)}")
                print(f"  Sample items inside zip: {namelist[:15]}")
    # only walk first 3 levels
    if root.count(os.sep) - bf_dir.count(os.sep) >= 2:
        dirs.clear()

print("\n=== C17 Directory ===")
for root, dirs, files in os.walk(c17_dir):
    print(f"Path: {root}")
    print(f"  Dirs ({len(dirs)}): {dirs[:10]}")
    print(f"  Files ({len(files)}): {files[:10]}")
    if root.count(os.sep) - c17_dir.count(os.sep) >= 2:
        dirs.clear()
