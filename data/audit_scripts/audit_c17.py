import os
import glob
from collections import defaultdict
from PIL import Image

c17_dir = r"C:\Users\DEEPTANU\Downloads\Human Bone Fracture C17 Dataset\Human Bone Fracture C17 Dataset"

subfolders = [d for d in os.listdir(c17_dir) if os.path.isdir(os.path.join(c17_dir, d))]
print(f"Total Subtypes in C17: {len(subfolders)}")

stats = defaultdict(lambda: {"count": 0, "sizes": [], "formats": set(), "modes": set(), "corrupt": 0})
total_imgs = 0
all_hashes = {}
duplicates = []

for folder in sorted(subfolders):
    fpath = os.path.join(c17_dir, folder)
    files = [f for f in os.listdir(fpath) if os.path.isfile(os.path.join(fpath, f))]
    for fname in files:
        total_imgs += 1
        img_path = os.path.join(fpath, fname)
        try:
            with Image.open(img_path) as img:
                stats[folder]["count"] += 1
                stats[folder]["sizes"].append(img.size)
                stats[folder]["formats"].add(img.format)
                stats[folder]["modes"].add(img.mode)
        except Exception as e:
            stats[folder]["corrupt"] += 1

print("\n--- C17 Subtype Distribution ---")
for folder in sorted(subfolders):
    s = stats[folder]
    sizes = s["sizes"]
    min_w = min(x[0] for x in sizes) if sizes else 0
    max_w = max(x[0] for x in sizes) if sizes else 0
    min_h = min(x[1] for x in sizes) if sizes else 0
    max_h = max(x[1] for x in sizes) if sizes else 0
    print(f"{folder:30s}: {s['count']:4d} images | formats: {s['formats']} | modes: {s['modes']} | size: ({min_w}x{min_h}) to ({max_w}x{max_h}) | corrupt: {s['corrupt']}")

print(f"\nTotal C17 Images Analyzed: {total_imgs}")
