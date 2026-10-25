# Local Lightweight Bone Fracture & Crack Detection Model

An ultra-lightweight, on-device diagnostic assistant for detecting and localizing bone fractures and subtle hairline cracks from radiographs (X-rays). Designed for local clinical decision support with **zero cloud dependencies**, **INT8 quantization**, calibrated confidence scoring, explicit uncertainty bands, and Grad-CAM explainability heatmaps.

---

## 🔬 Clinical Motivation & Design Philosophy

Missed fractures—particularly non-displaced fissures, hairline cracks, and scaphoid/femoral neck micro-trauma—represent one of the most common causes of diagnostic error in acute care.
- **Sensitivity-First Objective:** Missing a true fracture (False Negative) has far worse clinical consequences than a false positive. Operating thresholds prioritize high fracture recall ($\ge 0.35$).
- **Explainability Over Black-Box:** When a fracture is suspected, the system overlays a localized activation heatmap directly over cortical bone margins.
- **Explicit Uncertainty Band ($[0.35, 0.65]$):** Samples with borderline confidence are explicitly flagged as `UNCERTAIN (NEEDS HUMAN REVIEW)`, ensuring ambiguous or hairline presentations receive direct radiologist inspection.
- **Strict Data Integrity:** Evaluation partitions enforce strict **patient-level isolation** ($\text{Train} \cap \text{Test} = \emptyset$). No images from the same patient exist across partitions.

---

## ⚡ Hardware Constraints & Benchmarks

All benchmarks were measured locally on an **AMD Ryzen 9 8940HX (16 Cores / 32 Threads)** with **ONNX Runtime (CPU Execution Provider)**:

| Metric | Target Constraint | Measured Result (FP32) | Measured Result (INT8 Quantized) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Model File Size** | $< 25$ MB ($< 10$ MB INT8) | **12.27 MB** | **3.28 MB** | ✅ **Passed (67% below target)** |
| **CPU Latency (Mean)** | $< 300$ ms / image | **4.4 ms** | **95.0 ms** | ✅ **Passed (3x - 68x faster)** |
| **CPU Latency (P95)** | $< 400$ ms | **4.7 ms** | **102.7 ms** | ✅ **Passed** |
| **RAM Footprint** | $< 1.5$ GB | $\sim 280$ MB | $\sim 190$ MB | ✅ **Passed** |
| **Inference Dependency** | No PyTorch at runtime | `onnxruntime` only | `onnxruntime` only | ✅ **Passed** |
| **Offline Execution** | 100% Local | Zero API / Network calls | Zero API / Network calls | ✅ **Passed** |

---

## 📊 Datasets & Links

The model was developed, audited, and benchmarked on two complementary radiographic datasets:

### 1. Primary Clinical Training & Evaluation Dataset: BoneFract
- **Citation:** Noor, Raheel; Ali, Asghar (2026), *"BoneFract: A Bone Fracture Dataset"*, Mendeley Data, V1.
- **DOI & URL:** [doi: 10.17632/4cr7f2359x.1](https://data.mendeley.com/datasets/4cr7f2359x/1)
- **Source:** Peoples University of Medical & Health Sciences for Women (PUMHSW).
- **Size:** **47,931 unique patients** (exactly 1 study per patient).
- **Resolution:** Standardized $512 \times 512$ PNG radiographs.
- **Scope:** 7 anatomical regions: *Lower leg, Pelvis, Foot, Hand, Wrist, Arm, Thigh*.
- **Class Balance:** $70.9\%$ Normal (Negative) vs. $29.1\%$ Fracture (Positive).
- **Split Breakdown:** Train (38,344 / 80%), Validation (4,793 / 10%), Locked Test (4,794 / 10%). Zero patient leakage.

### 2. Subtle & Subtype Stress Benchmark: Human Bone Fracture C17
- **Citation:** Sarker, Shatabdi; Roy, Avizit; Sharmin, Shaila; Rahman, Shakila; Uddin, Jia (2025), *"Human Bone Fracture C17 Dataset"*, Mendeley Data, V1.
- **DOI & URL:** [doi: 10.17632/2j8vvz3j6v.1](https://data.mendeley.com/datasets/2j8vvz3j6v/1)
- **Size:** 2,191 images across **17 specific fracture subtypes**.
- **Role:** Out-of-distribution benchmark targeting challenging subtle presentations:
  - *Hairline Fractures* (139 images)
  - *Stress Fractures* (147 images)
  - *Greenstick Fractures* (136 images)
  - *Avulsion Fractures* (141 images)
  - *Impacted & Longitudinal Cracks*

---

## 📁 Repository Structure

```
bone_fracture_detection/
├── configs/
│   └── config.yaml                     # Pipeline configuration (data, loss, thresholds)
├── models/
│   ├── bone_fracture_detector.onnx     # Exported FP32 ONNX model (12.3 MB)
│   └── bone_fracture_detector_int8.onnx# Quantized INT8 ONNX model (3.28 MB)
├── reports/
│   ├── data_audit_report.md            # Detailed clinical data audit
│   ├── export_benchmark.json           # Hardware latency & size verification
│   └── inference_overlay.png           # Sample crack localization heatmap
├── src/
│   ├── dataset.py                      # Zip-streaming dataset loader & radiograph normalization
│   ├── model.py                        # MobileNetV3-Large with Grad-CAM hooking
│   ├── train.py                        # Resumable training loop with class-weighted loss
│   ├── evaluate.py                     # Locked test evaluation, ECE, & subtype benchmarks
│   ├── export_onnx.py                  # ONNX export, INT8 quantization & benchmarking
│   └── infer.py                        # Standalone ONNX Runtime inference CLI
├── app.py                              # Interactive local Gradio web application
├── requirements.txt                    # Pinned dependencies
├── .gitignore                          # Clean repository hygiene
└── README.md                           # Documentation & user guide
```

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install requirements in Python 3.10+:
```bash
git clone <YOUR_REPO_URL>
cd bone_fracture_detection
pip install -r requirements.txt
```

### 2. Standalone CLI Inference (Pure ONNX Runtime)
Run diagnostic inference on any radiograph without needing PyTorch:
```bash
python src/infer.py --image path/to/xray.jpg --overlay reports/overlay.png
```
**Sample Output:**
```text
========================================================================
 NOTICE: Research / Assistive Tool. Not a Certified Medical Device.
 Not a substitute for professional radiologist evaluation.
========================================================================

--- Diagnostic Inference Result ---
File:           sample_wrist_xray.png
Status:         FRACTURE / CRACK DETECTED
Confidence:     89.4%
Human Review:   No
Overlay Saved:  reports/overlay.png
```

### 3. Launch Local Web Application
Launch the interactive drag-and-drop diagnostic interface:
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your web browser. Features include:
- Drag-and-drop radiograph viewer
- Adjustable clinical sensitivity sliders
- Color-coded diagnostic badge (**Green**: Normal, **Red**: Fracture, **Amber**: Human Review Needed)
- Side-by-side crack localization heatmap overlay

---

## 🛡️ Regulatory & Clinical Disclaimer

> **IMPORTANT NOTICE:**  
> This software is intended strictly as a **research and clinical triage assistive tool**. It is **not** an FDA-cleared or CE-marked medical device and has not been certified for primary diagnostic interpretations. It is **not** a substitute for evaluation by a qualified medical specialist or radiologist.
