# Comprehensive Data Audit Report: Bone Fracture & Crack Detection

**Date:** 2026-10-25  
**Datasets Inspected:**
1. **Primary Dataset:** BoneFract (`C:\Users\DEEPTANU\Downloads\BoneFract A Bone Fracture Dataset`)
2. **Subtype & Subtle Crack Benchmark:** Human Bone Fracture C17 (`C:\Users\DEEPTANU\Downloads\Human Bone Fracture C17 Dataset`)

---

## 1. Primary Dataset: BoneFract

### 1.1 Structural & File Characteristics
- **Format:** High-resolution PNG images, paired with detailed Excel clinical index (`BoneFract_Xray_Dataset_Sheets.xlsx`).
- **Archive Size:** 3.68 GB compressed ZIP containing **47,931 images** across 143,819 filesystem nodes.
- **Resolution:** Standardized **512 × 512 pixels** (exact 1.00 square aspect ratio across all studies).
- **Color Depth & Channels:** 8-bit per channel; stored as RGB and Grayscale (L), with pixel intensity dynamic range $[0, 255]$ and mean background intensity of $58.24$.
- **Corrupt Files:** $0$ corrupt files detected in the raw archive.

### 1.2 Annotation & Localization Modality
- **Label Hierarchy:** Folder and Excel study-level binary classification: `Negative` (Normal, no fracture: 0) vs. `Positive` (Fracture present: 1).
- **Bounding Boxes / Masks:** **None present** in the dataset metadata.
- **Localization Strategy:** Fully supported via **Grad-CAM explainability heatmaps** computed on the final convolutional feature maps. This enables highlighting hairline crack lines, displacement margins, and cortical disruptions without requiring manual bounding-box annotations.

### 1.3 Patient Isolation & Leakage Audit
- **Unique Patients:** **47,931 unique patients** across 47,931 total studies (exactly 1 study/image per patient).
- **Partition Verification:**
  - `train` set: 38,344 patients (80.0%)
  - `valid` set: 4,793 patients (10.0%)
  - `test` set: 4,794 patients (10.0%)
- **Data Leakage Check:**
  - $\text{Train} \cap \text{Validation}$ patient overlap: **0 patients**
  - $\text{Train} \cap \text{Test}$ patient overlap: **0 patients**
  - $\text{Validation} \cap \text{Test}$ patient overlap: **0 patients**
- **Conclusion:** Zero patient-level leakage exists. The evaluation splits represent true out-of-sample patient generalization.

### 1.4 Class Balance & Anatomical Breakdown
- **Overall Class Balance:**
  - Negative (No Fracture): **70.9%** ($34,010$ samples)
  - Positive (Fracture): **29.1%** ($13,921$ samples)
  - Natural class imbalance ratio: **~2.44 : 1**. Will be addressed via focal loss / positive class weighting ($w \approx 2.44$).
- **Distribution across 7 Anatomical Regions:**
  | Anatomical Region | Total Studies | Fracture Rate (%) | Clinical Relevance |
  | :--- | :--- | :--- | :--- |
  | **Lower Leg** (Tibia / Fibula) | 10,996 | 36.8% | High incidence of sports/trauma fractures |
  | **Pelvis** | 6,983 | 17.9% | Complex ring anatomy, subtle pubic rami fractures |
  | **Foot** (Tarsals / Metatarsals) | 6,840 | 28.0% | Subtle stress fractures, avulsions (5th metatarsal) |
  | **Hand** (Phalanges / Metacarpals) | 6,032 | 17.9% | Fine hairline cracks, tuft fractures |
  | **Wrist** (Distal Radius / Scaphoid) | 6,019 | 31.5% | Colles fractures, scaphoid hairline fissures |
  | **Arm** (Humerus / Forearm) | 5,598 | 17.4% | Shaft fractures, supracondylar cracks |
  | **Thigh** (Femur / Hip) | 5,463 | 50.8% | Intertrochanteric, femoral neck trauma |

---

## 2. Secondary Benchmark: Human Bone Fracture C17

- **Total Images:** 2,191 images across **17 specific fracture subtypes**.
- **Role:** Out-of-distribution external benchmark specifically targeted at subtle, hairline, and pathological fracture evaluation.
- **Subtype Inventory:**
  - **Hairline Fracture:** 139 images
  - **Stress Fracture:** 147 images (1 corrupted file identified and filtered)
  - **Greenstick Fracture:** 136 images (pediatric incomplete fractures)
  - **Impacted Fracture:** 161 images
  - **Avulsion Fracture:** 141 images
  - **Compression-Crush Fracture:** 150 images
  - **Fracture Dislocation:** 159 images
  - **Spiral Fracture:** 134 images
  - **Longitudinal Fracture:** 129 images
  - **Pathological Fracture:** 129 images
  - **Oblique Fracture:** 124 images
  - **Closed (Simple) Fracture:** 112 images
  - **Intra-articular Fracture:** 104 images
  - **Open (Compound) Fracture:** 86 images
  - **Segmental Fracture:** 74 images
  - **Transverse Fracture:** 47 images
  - **Comminuted Fracture:** 218 images

---

## 3. Potential Bias, Artifacts & Shortcut Learning Analysis

1. **Burned-In Orientation Markers & Text:**
   - Perceptual scan of 1,000 radiograph samples revealed **18.9%** containing high-contrast corner annotations (`L` / `R` anatomical side markers, technician lead initials, hospital stamps).
   - **Mitigation:**
     - Training augmentations including mild random shifts and corner/border masking.
     - Saliency and Grad-CAM inspection during evaluation to ensure gradient activations localize on bone cortex and trabecular architecture rather than peripheral markers.
2. **Perceptual Near-Duplicates:**
   - Hamming distance ($d \le 2$) dHash analysis yielded $<0.1\%$ near-duplicate pairs across different patients, confirming exceptional radiographic diversity.
3. **Splints & Hardware:**
   - A minority of acute radiographs show external immobilization (casts, radiopaque splints). Weighted sampling and augmentation ensure the backbone learns osseous discontinuity rather than orthopedic appliances.

---

## 4. Pipeline Decision & Execution Strategy

1. **Locked Splits:**
   - Use the strict patient-stratified splits: **Train (80%, 38,344)**, **Validation (10%, 4,793)**, and **Locked Test (10%, 4,794)**.
   - The test set is permanently locked for final post-quantization verification only.
2. **External Subtle Benchmark:**
   - The 2,191 C17 images (specifically the 139 Hairline and 147 Stress cracks) will serve as an independent stress-test set to evaluate sensitivity on difficult low-contrast hairline fractures.
