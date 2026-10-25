"""
Dataset module for Bone Fracture & Crack Detection.
Supports zero-overhead direct streaming from the BoneFract ZIP archive
and directory-based loading for the Human Bone Fracture C17 benchmark.
"""

import os
import io
import zipfile
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class BoneFractDataset(Dataset):
    """
    BoneFract Dataset loaded directly from the zip archive using the Excel metadata.
    Prevents uncompressing 47k files to disk, eliminating filesystem overhead.
    """
    def __init__(self, zip_path, excel_path, split="train", img_size=384, transform=None):
        self.zip_path = zip_path
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.zf = None  # Lazy zipfile per process/worker
        
        # Load metadata
        df = pd.read_excel(excel_path, sheet_name=f"{split}_labeled_studies", header=None)
        df.columns = ["study_path", "label"]
        
        # Parse body parts and image paths
        # study_path format: BoneFract A Bone Fracture Dataset/{split}/{body_part}/{patient_id}/{label_str}/
        def parse_row(p):
            parts = p.strip("/").split("/")
            body_part = parts[2] if len(parts) > 2 else "Unknown"
            patient_id = parts[3] if len(parts) > 3 else "Unknown"
            label_str = parts[4] if len(parts) > 4 else "Negative"
            return body_part, patient_id, label_str
            
        parsed = df["study_path"].apply(parse_row)
        df["body_part"] = [x[0] for x in parsed]
        df["patient_id"] = [x[1] for x in parsed]
        df["label_str"] = [x[2] for x in parsed]
        
        # Construct internal image path:
        # e.g. BoneFract A Bone Fracture Dataset/test/Arm/patient03647/Negative/Arm_patient03647_Negative_001.png
        df["img_path"] = df.apply(
            lambda r: f"{r['study_path'].strip('/')}/{r['body_part']}_{r['patient_id']}_{r['label_str']}_001.png",
            axis=1
        )
        self.samples = df.to_dict("records")
        
    def _get_zip(self):
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path, "r")
        return self.zf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        img_path = row["img_path"]
        label = float(row["label"])
        body_part = row["body_part"]
        
        zf = self._get_zip()
        try:
            img_bytes = zf.read(img_path)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except KeyError:
            # Fallback if filename variant
            prefix = row["study_path"].strip("/")
            candidates = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".png")]
            if candidates:
                img_bytes = zf.read(candidates[0])
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                img = Image.new("RGB", (self.img_size, self.img_size), color=0)
                
        # Radiograph preprocessing: intensity contrast enhancement (percentile normalization)
        arr = np.array(img, dtype=np.float32)
        p2, p98 = np.percentile(arr, (2, 98))
        if p98 > p2:
            arr = np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            
        if self.transform:
            img = self.transform(img)
        else:
            default_t = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img = default_t(img)
            
        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.float32),
            "body_part": body_part,
            "patient_id": row["patient_id"],
            "path": img_path
        }


class C17SubtypeDataset(Dataset):
    """
    Human Bone Fracture C17 benchmark for evaluating fracture subtypes,
    specifically hairline, stress, greenstick, and subtle cracks.
    """
    def __init__(self, root_dir, img_size=384, transform=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.samples = []
        
        subfolders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for folder in sorted(subfolders):
            fpath = os.path.join(root_dir, folder)
            for fname in os.listdir(fpath):
                f_full = os.path.join(fpath, fname)
                if os.path.isfile(f_full) and not fname.endswith(".avif"):  # filter unsupported formats
                    self.samples.append({
                        "path": f_full,
                        "subtype": folder,
                        "label": 1.0  # All C17 samples are verified fractures
                    })
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        try:
            img = Image.open(row["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), color=0)
            
        if self.transform:
            img = self.transform(img)
        else:
            default_t = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img = default_t(img)
            
        return {
            "image": img,
            "label": torch.tensor(row["label"], dtype=torch.float32),
            "subtype": row["subtype"],
            "path": row["path"]
        }


def get_transforms(img_size=384, is_train=True):
    """
    Diagnostic-safe radiograph augmentations.
    Prevents heavy distortive transforms that fabricate or erase hairline cracks.
    """
    if is_train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
