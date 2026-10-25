"""
Local Web Application for Bone Fracture & Crack Detection.
Built with Gradio for drag-and-drop local diagnostic assistance.
Runs 100% locally on localhost via ONNX Runtime.
"""

import os
import sys
import numpy as np
from PIL import Image
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.infer import FractureInferenceEngine, create_overlay, preprocess_image, DISCLAIMER

# Initialize Engine
MODEL_PATH = "models/bone_fracture_detector_int8.onnx"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/bone_fracture_detector.onnx"

engine = None
try:
    engine = FractureInferenceEngine(model_path=MODEL_PATH, threshold_low=0.35, threshold_high=0.65)
except Exception as e:
    print(f"Warning: ONNX model not found yet ({e}). Engine will initialize once exported.")

def analyze_xray(input_img, thresh_low=0.35, thresh_high=0.65):
    if input_img is None:
        return None, None, "Please upload or select an X-ray image."
        
    global engine
    if engine is None or not os.path.exists(engine.model_path):
        return None, None, "Model file not found. Please run export first: python src/export_onnx.py"
        
    # Convert numpy array to PIL Image
    if isinstance(input_img, np.ndarray):
        pil_img = Image.fromarray(input_img).convert("RGB")
    else:
        pil_img = input_img.convert("RGB")
        
    temp_path = "temp_input.png"
    pil_img.save(temp_path)
    
    engine.threshold_low = thresh_low
    engine.threshold_high = thresh_high
    res = engine.predict(temp_path)
    
    prob = res["probability"]
    prob_pct = f"{prob * 100:.1f}%"
    
    # Decision formatting
    if res["decision_code"] == "FRACTURE":
        status_md = (
            f"### <span style='color:#e74c3c;'>⚠️ FRACTURE / CRACK DETECTED</span>\n\n"
            f"**Fracture Probability:** `{prob_pct}` (Above high-sensitivity threshold of `{thresh_low * 100:.0f}%`)\n\n"
            f"**Action Recommended:** Osseous discontinuity or crack pattern identified. Refer for orthopedic assessment."
        )
    elif res["decision_code"] == "NORMAL":
        status_md = (
            f"### <span style='color:#2ecc71;'>✅ NO FRACTURE DETECTED</span>\n\n"
            f"**Fracture Probability:** `{prob_pct}` (Below clinical alert threshold of `{thresh_low * 100:.0f}%`)\n\n"
            f"**Action Recommended:** Cortical margins appear intact."
        )
    else:
        status_md = (
            f"### <span style='color:#f39c12;'>⚠️ UNCERTAIN - HUMAN REVIEW REQUIRED</span>\n\n"
            f"**Fracture Probability:** `{prob_pct}` (Falls inside borderline confidence band `[{thresh_low * 100:.0f}%, {thresh_high * 100:.0f}%]`)\n\n"
            f"**Action Recommended:** Subtle or ambiguous cortical line detected. Mandatory human radiologist review required."
        )
        
    overlay = create_overlay(pil_img, prob)
    return pil_img, overlay, status_md

def launch_app():
    custom_css = """
    .disclaimer-box {
        background-color: #fcf8e3;
        border: 1px solid #faebcc;
        color: #8a6d3b;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 15px;
        font-size: 0.9em;
    }
    """
    
    with gr.Blocks(title="Local Bone Fracture & Crack Detector", css=custom_css) as demo:
        gr.Markdown(
            """
            # 🦴 Local Lightweight Bone Fracture & Crack Detection
            ### On-Device Radiograph Diagnostic Assistant (<10 MB INT8, <300ms CPU Inference)
            """
        )
        
        gr.HTML(
            f"""
            <div class="disclaimer-box">
                <strong>⚠️ Regulatory Notice:</strong> Research and clinical triage assistive tool. Not an FDA/CE certified medical device. 
                Not a replacement for qualified physician review.
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload Radiograph (PNG, JPG, DICOM-converted)")
                
                with gr.Accordion("Clinical Sensitivity Settings", open=False):
                    thresh_low = gr.Slider(0.1, 0.5, value=0.35, step=0.05, label="High-Sensitivity Alert Threshold (Recall-focused)")
                    thresh_high = gr.Slider(0.5, 0.9, value=0.65, step=0.05, label="High-Specificity Threshold (Confident Fracture)")
                    
                analyze_btn = gr.Button("🔍 Analyze Radiograph", variant="primary")
                
            with gr.Column(scale=1):
                result_markdown = gr.Markdown("### Diagnostic Assessment\n*Upload an image to view results.*")
                with gr.Row():
                    out_orig = gr.Image(type="pil", label="Preprocessed X-Ray")
                    out_overlay = gr.Image(type="pil", label="Crack Localization Heatmap")
                    
        analyze_btn.click(
            fn=analyze_xray,
            inputs=[input_image, thresh_low, thresh_high],
            outputs=[out_orig, out_overlay, result_markdown]
        )
        
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

if __name__ == "__main__":
    launch_app()
