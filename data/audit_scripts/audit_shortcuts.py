import zipfile
import io
import random
from PIL import Image
import numpy as np

def dhash(image, hash_size=8):
    # Grayscale and resize to (hash_size + 1, hash_size)
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image)
    # Compare adjacent pixels
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()

def hamming_dist(h1, h2):
    return np.count_nonzero(h1 != h2)

zip_path = r"C:\Users\DEEPTANU\Downloads\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset.zip"

with zipfile.ZipFile(zip_path, 'r') as zf:
    img_files = [f for f in zf.namelist() if f.endswith('.png')]
    random.seed(42)
    sample_files = random.sample(img_files, 1000)
    
    hashes = []
    border_marker_count = 0
    
    for f in sample_files:
        data = zf.read(f)
        with Image.open(io.BytesIO(data)) as img:
            h = dhash(img)
            hashes.append((f, h))
            
            # Check for high-contrast small markers in image corners (typical for 'L' / 'R' / hospital labels)
            arr = np.array(img.convert('L'))
            # Check top 40 rows and bottom 40 rows corners (40x40 corners)
            corners = [
                arr[:40, :40], arr[:40, -40:],
                arr[-40:, :40], arr[-40:, -40:]
            ]
            has_marker = any((c > 200).sum() > 25 and (c < 50).sum() > 50 for c in corners)
            if has_marker:
                border_marker_count += 1

print(f"Sample size: {len(sample_files)}")
print(f"Images with detected corner markers/annotations: {border_marker_count} / {len(sample_files)} ({border_marker_count/len(sample_files)*100:.1f}%)")

# Check for identical or near-identical images (Hamming distance <= 2)
near_dupes = 0
for i in range(len(hashes)):
    for j in range(i + 1, min(i + 50, len(hashes))): # check neighborhood
        dist = hamming_dist(hashes[i][1], hashes[j][1])
        if dist <= 2:
            near_dupes += 1

print(f"Near-duplicate pairs found in random sample: {near_dupes}")
