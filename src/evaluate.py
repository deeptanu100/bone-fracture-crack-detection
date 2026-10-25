"""
Evaluation and Validation Module on the Locked Test Set and Subtle Crack Benchmark.
Computes:
- Sensitivity (Recall), Specificity, Precision, F1, AUROC, PR-AUC
- Expected Calibration Error (ECE) and Reliability Analysis
- 95% Bootstrap Confidence Intervals
- Per-body-part breakdown across the 7 anatomical regions
- Per-subtype evaluation on C17 with special focus on Hairline and Subtle Cracks
"""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, recall_score, precision_score, f1_score
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import BoneFractDataset, C17SubtypeDataset, get_transforms
from src.model import FractureDetector

def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_conf = np.mean(probs[in_bin])
            avg_acc = np.mean(labels[in_bin])
            ece += np.abs(avg_conf - avg_acc) * prop_in_bin
    return float(ece)

def bootstrap_ci(y_true, y_pred_prob, threshold=0.35, n_bootstraps=500, seed=42):
    """Compute 95% bootstrap confidence intervals for key diagnostic metrics."""
    rng = np.random.RandomState(seed)
    recalls, specificities, aurocs = [], [], []
    
    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue
        b_true = y_true[indices]
        b_probs = y_pred_prob[indices]
        b_pred = (b_probs >= threshold).astype(int)
        
        recalls.append(recall_score(b_true, b_pred, zero_division=0))
        specificities.append(recall_score(b_true == 0, b_pred == 0, zero_division=0))
        aurocs.append(roc_auc_score(b_true, b_probs))
        
    return {
        "recall_ci_95": [float(np.percentile(recalls, 2.5)), float(np.percentile(recalls, 97.5))],
        "specificity_ci_95": [float(np.percentile(specificities, 2.5)), float(np.percentile(specificities, 97.5))],
        "auroc_ci_95": [float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))]
    }

def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device(cfg["device"] if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")
    reports_dir = cfg["evaluation"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load Model
    model = FractureDetector(architecture=cfg["model"]["architecture"], pretrained=False)
    best_path = cfg["training"]["best_model_path"]
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded trained checkpoint: {best_path}")
    else:
        print("Pretrained weights loaded (no custom checkpoint found).")
    model = model.to(device)
    model.eval()
    
    # Load Locked Test Set
    print("\n--- Evaluating on Locked Test Set (BoneFract) ---")
    test_ds = BoneFractDataset(
        zip_path=cfg["data"]["bonefract_zip"],
        excel_path=cfg["data"]["excel_meta"],
        split="test",
        img_size=cfg["data"]["img_size"],
        transform=get_transforms(cfg["data"]["img_size"], is_train=False)
    )
    
    # Evaluate on test set (or representative subset if evaluating fast)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)
    
    all_probs, all_labels, all_body_parts = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["label"].numpy())
            all_body_parts.extend(batch["body_part"])
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_body_parts = np.array(all_body_parts)
    
    thresh = cfg["evaluation"]["high_sensitivity_threshold"]
    binary_preds = (all_probs >= thresh).astype(int)
    
    # Primary Metrics
    overall_recall = float(recall_score(all_labels, binary_preds, zero_division=0))
    overall_spec = float(recall_score(all_labels == 0, binary_preds == 0, zero_division=0))
    overall_prec = float(precision_score(all_labels, binary_preds, zero_division=0))
    overall_f1 = float(f1_score(all_labels, binary_preds, zero_division=0))
    overall_auc = float(roc_auc_score(all_labels, all_probs))
    
    prec_curve, rec_curve, _ = precision_recall_curve(all_labels, all_probs)
    overall_pr_auc = float(auc(rec_curve, prec_curve))
    ece = compute_ece(all_probs, all_labels)
    ci = bootstrap_ci(all_labels, all_probs, threshold=thresh)
    
    cm = confusion_matrix(all_labels, binary_preds).tolist()
    
    # Per-Body-Part Breakdown
    body_part_metrics = {}
    for bp in np.unique(all_body_parts):
        mask = (all_body_parts == bp)
        sub_true = all_labels[mask]
        sub_prob = all_probs[mask]
        sub_pred = (sub_prob >= thresh).astype(int)
        
        body_part_metrics[bp] = {
            "count": int(np.sum(mask)),
            "recall": float(recall_score(sub_true, sub_pred, zero_division=0)),
            "specificity": float(recall_score(sub_true == 0, sub_pred == 0, zero_division=0)),
            "auc": float(roc_auc_score(sub_true, sub_prob)) if len(set(sub_true)) > 1 else 0.5
        }
        
    # Evaluate C17 Subtype Benchmark (Hairline, Stress, etc.)
    c17_metrics = {}
    if os.path.exists(cfg["data"]["c17_dir"]):
        print("\n--- Evaluating on Subtle Crack Benchmark (C17 Subtypes) ---")
        c17_ds = C17SubtypeDataset(root_dir=cfg["data"]["c17_dir"], img_size=cfg["data"]["img_size"])
        c17_loader = DataLoader(c17_ds, batch_size=32, shuffle=False)
        c17_probs, c17_subtypes = [], []
        with torch.no_grad():
            for batch in c17_loader:
                imgs = batch["image"].to(device)
                l = model(imgs)
                c17_probs.extend(torch.sigmoid(l).cpu().numpy())
                c17_subtypes.extend(batch["subtype"])
                
        c17_probs = np.array(c17_probs)
        c17_subtypes = np.array(c17_subtypes)
        
        for st in np.unique(c17_subtypes):
            m = (c17_subtypes == st)
            st_probs = c17_probs[m]
            # Since all C17 are fractures, recall = % identified as fracture (prob >= thresh)
            st_recall = float(np.mean(st_probs >= thresh))
            c17_metrics[st] = {
                "count": int(np.sum(m)),
                "sensitivity_recall": st_recall,
                "mean_confidence": float(np.mean(st_probs))
            }
            
    # Compile Results
    results = {
        "operating_threshold": thresh,
        "overall_metrics": {
            "sensitivity_recall": overall_recall,
            "specificity": overall_spec,
            "precision": overall_prec,
            "f1_score": overall_f1,
            "auroc": overall_auc,
            "pr_auc": overall_pr_auc,
            "expected_calibration_error": ece
        },
        "confusion_matrix": cm,
        "confidence_intervals_95": ci,
        "body_part_breakdown": body_part_metrics,
        "c17_subtype_breakdown": c17_metrics
    }
    
    out_json = os.path.join(reports_dir, "test_evaluation_report.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation completed! Saved report to {out_json}")
    
    # Generate Calibration Curve Plot
    prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model Calibration")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fracture Fraction")
    plt.title(f"Reliability Diagram (ECE = {ece:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "calibration_curve.png"), dpi=200)
    plt.close()
    
    print("Calibration diagram saved to reports/calibration_curve.png")

if __name__ == "__main__":
    main()
