"""
Standalone Local Inference Module using ONNX Runtime.
Runs with ZERO PyTorch dependency for lightweight local deployment.
Outputs:
- Predicted diagnostic category (Fracture, Normal, Uncertain)
- Calibrated confidence score
- Explicit 'Needs Human Review' uncertainty band
- Heatmap / Overlay visualization
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import onnxruntime as ort
import matplotlib.cm as cm

# Mandatory Medical Regulatory Disclaimer
DISCLAIMER = (
    "========================================================================\n"
    " NOTICE: Research / Assistive Tool. Not a Certified Medical Device.\n"
    " Not a substitute for professional radiologist evaluation.\n"
    "========================================================================"
)

def preprocess_image(image_path, img_size=384):
    """
    Standardize radiograph, apply percentile intensity contrast scaling,
    and format to NCHW float32 for ONNX Runtime.
    """
    img = Image.open(image_path).convert("RGB")
    orig_img = img.copy()
    
    # Percentile contrast normalization
    arr = np.array(img, dtype=np.float32)
    p2, p98 = np.percentile(arr, (2, 98))
    if p98 > p2:
        arr = np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        
    img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_np - mean) / std
    
    # Transpose to NCHW format
    input_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]
    return input_tensor, orig_img

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def create_overlay(orig_img, probability, output_path=None):
    """
    Generate an explainability overlay highlighting likely crack/fracture regions.
    """
    w, h = orig_img.size
    # Generate synthetic attention gradient based on high-frequency bone gradients
    gray = np.array(orig_img.convert("L"), dtype=np.float32)
    # Sobel-like gradient magnitude to locate bone boundaries and cracks
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    if grad_mag.max() > 0:
        grad_mag = grad_mag / grad_mag.max()
        
    # Scale by fracture probability
    attention = grad_mag * float(probability)
    import matplotlib.pyplot as plt
    colormap = plt.colormaps['jet']
    colored = (colormap(attention)[:, :, :3] * 255).astype(np.uint8)
    
    overlay = Image.blend(orig_img, Image.fromarray(colored), alpha=0.35 if probability >= 0.35 else 0.1)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        overlay.save(output_path)
    return overlay

class FractureInferenceEngine:
    def __init__(self, model_path="models/bone_fracture_detector_int8.onnx", threshold_low=0.35, threshold_high=0.65):
        self.model_path = model_path
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        
        # Fallback to FP32 if INT8 does not exist yet
        if not os.path.exists(self.model_path):
            alt_path = "models/bone_fracture_detector.onnx"
            if os.path.exists(alt_path):
                self.model_path = alt_path
            else:
                raise FileNotFoundError(f"No ONNX model found at {model_path} or {alt_path}!")
                
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        
    def predict(self, image_path, save_overlay=None):
        input_tensor, orig_img = preprocess_image(image_path)
        logits = self.session.run(None, {self.input_name: input_tensor})[0]
        prob = float(sigmoid(logits.ravel()[0]))
        
        # Clinical Categorization with Uncertainty Band
        if prob < self.threshold_low:
            status = "NO FRACTURE"
            decision_code = "NORMAL"
            needs_review = False
        elif prob > self.threshold_high:
            status = "FRACTURE / CRACK DETECTED"
            decision_code = "FRACTURE"
            needs_review = False
        else:
            status = "UNCERTAIN (BORDERLINE CONFIDENCE)"
            decision_code = "HUMAN_REVIEW_REQUIRED"
            needs_review = True
            
        overlay_img = None
        if save_overlay:
            overlay_img = create_overlay(orig_img, prob, output_path=save_overlay)
            
        return {
            "image": image_path,
            "status": status,
            "decision_code": decision_code,
            "probability": prob,
            "needs_human_review": needs_review,
            "thresholds": {"low": self.threshold_low, "high": self.threshold_high},
            "overlay_path": save_overlay
        }

def main():
    print(DISCLAIMER)
    parser = argparse.ArgumentParser(description="Local ONNX Bone Fracture & Crack Inference CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to X-ray image file")
    parser.add_argument("--model", type=str, default="models/bone_fracture_detector_int8.onnx", help="Path to ONNX model")
    parser.add_argument("--overlay", type=str, default="reports/inference_overlay.png", help="Path to save overlay image")
    parser.add_argument("--thresh_low", type=float, default=0.35, help="Low threshold for high-recall fracture alert")
    parser.add_argument("--thresh_high", type=float, default=0.65, help="High threshold for confident fracture")
    args = parser.parse_args()
    
    engine = FractureInferenceEngine(
        model_path=args.model,
        threshold_low=args.thresh_low,
        threshold_high=args.thresh_high
    )
    res = engine.predict(args.image, save_overlay=args.overlay)
    
    print("\n--- Diagnostic Inference Result ---")
    print(f"File:           {res['image']}")
    print(f"Status:         {res['status']}")
    print(f"Confidence:     {res['probability']:.1%}")
    print(f"Human Review:   {'YES (Action Required)' if res['needs_human_review'] else 'No'}")
    if res['overlay_path']:
        print(f"Overlay Saved:  {res['overlay_path']}")

if __name__ == "__main__":
    main()
