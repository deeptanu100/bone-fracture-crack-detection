import zipfile
import io
import random
from PIL import Image
import numpy as np

zip_path = r"C:\Users\DEEPTANU\Downloads\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset.zip"

print("Inspecting BoneFract image files directly from zip...")
with zipfile.ZipFile(zip_path, 'r') as zf:
    img_files = [f for f in zf.namelist() if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
    print(f"Total image files found in zip: {len(img_files)}")
    
    random.seed(42)
    sample_files = random.sample(img_files, 200)
    
    sizes = []
    modes = set()
    formats = set()
    aspect_ratios = []
    min_intensities = []
    max_intensities = []
    mean_intensities = []
    
    for f in sample_files:
        data = zf.read(f)
        with Image.open(io.BytesIO(data)) as img:
            sizes.append(img.size)
            modes.add(img.mode)
            formats.add(img.format)
            aspect_ratios.append(img.width / img.height)
            arr = np.array(img)
            min_intensities.append(arr.min())
            max_intensities.append(arr.max())
            mean_intensities.append(arr.mean())

    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    
    print("\n--- Image Properties Sample (N=200) ---")
    print("Formats:", formats)
    print("Modes:", modes)
    print(f"Width: min={min(widths)}, median={int(np.median(widths))}, max={max(widths)}")
    print(f"Height: min={min(heights)}, median={int(np.median(heights))}, max={max(heights)}")
    print(f"Aspect ratio: min={min(aspect_ratios):.2f}, median={np.median(aspect_ratios):.2f}, max={max(aspect_ratios):.2f}")
    print(f"Intensity Range: min={min(min_intensities)}, max={max(max_intensities)}, avg mean={np.mean(mean_intensities):.2f}")
