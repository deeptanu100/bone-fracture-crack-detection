"""
Model Export, Verification, and Quantization Module.
Exports PyTorch weights to:
1. Standard ONNX (FP32)
2. Quantized ONNX (INT8 Dynamic)
Benchmarks latency, RAM, and numerical fidelity against the user constraints:
- File size < 25 MB (< 10 MB INT8)
- CPU Latency < 300 ms
- Local onnxruntime inference with zero PyTorch dependency
"""

import os
import sys
import time
import json
import yaml
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FractureDetector

def export_and_quantize():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    onnx_fp32_path = cfg["export"]["onnx_fp32_path"]
    onnx_int8_path = cfg["export"]["onnx_int8_path"]
    reports_dir = cfg["evaluation"]["reports_dir"]
    os.makedirs(os.path.dirname(onnx_fp32_path), exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    # Load PyTorch Model
    print("Loading PyTorch model for ONNX export...")
    model = FractureDetector(architecture=cfg["model"]["architecture"], pretrained=True)
    best_path = cfg["training"]["best_model_path"]
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location="cpu"))
        print(f"Loaded trained checkpoint: {best_path}")
    model.eval()
    
    # Remove hook handles for clean static graph export
    for h in model._hook_handles:
        h.remove()
    model._hook_handles.clear()
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, cfg["data"]["img_size"], cfg["data"]["img_size"])
    print(f"Exporting FP32 ONNX to {onnx_fp32_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_fp32_path,
        export_params=True,
        dynamo=False,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}}
    )
    
    # Verify ONNX model graph
    onnx_model = onnx.load(onnx_fp32_path)
    onnx.checker.check_model(onnx_model)
    fp32_size_mb = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    print(f"FP32 ONNX Model successfully validated! Size: {fp32_size_mb:.2f} MB")
    
    # INT8 Dynamic Quantization
    print(f"Applying INT8 Dynamic Quantization to {onnx_int8_path}...")
    quantize_dynamic(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        weight_type=QuantType.QInt8
    )
    int8_size_mb = os.path.getsize(onnx_int8_path) / (1024 * 1024)
    print(f"INT8 Quantized Model saved! Size: {int8_size_mb:.2f} MB")
    
    # Numerical Consistency Check
    session_fp32 = ort.InferenceSession(onnx_fp32_path, providers=["CPUExecutionProvider"])
    session_int8 = ort.InferenceSession(onnx_int8_path, providers=["CPUExecutionProvider"])
    
    np_input = dummy_input.numpy()
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()
    ort_fp32_out = session_fp32.run(None, {"input": np_input})[0]
    ort_int8_out = session_int8.run(None, {"input": np_input})[0]
    
    fp32_diff = float(np.max(np.abs(pt_out - ort_fp32_out)))
    int8_diff = float(np.max(np.abs(pt_out - ort_int8_out)))
    print(f"Numerical Max Absolute Difference (PyTorch vs ONNX FP32): {fp32_diff:.6e}")
    print(f"Numerical Max Absolute Difference (PyTorch vs ONNX INT8): {int8_diff:.6e}")
    
    # Latency Benchmarks on CPU
    print("\nBenchmarking CPU inference latency (50 iterations)...")
    latencies_fp32 = []
    for _ in range(50):
        t0 = time.perf_counter()
        session_fp32.run(None, {"input": np_input})
        latencies_fp32.append((time.perf_counter() - t0) * 1000.0)
        
    latencies_int8 = []
    for _ in range(50):
        t0 = time.perf_counter()
        session_int8.run(None, {"input": np_input})
        latencies_int8.append((time.perf_counter() - t0) * 1000.0)
        
    mean_fp32 = float(np.mean(latencies_fp32))
    p95_fp32 = float(np.percentile(latencies_fp32, 95))
    mean_int8 = float(np.mean(latencies_int8))
    p95_int8 = float(np.percentile(latencies_int8, 95))
    
    print(f"ONNX FP32 Latency: Mean = {mean_fp32:.1f} ms | P95 = {p95_fp32:.1f} ms")
    print(f"ONNX INT8 Latency: Mean = {mean_int8:.1f} ms | P95 = {p95_int8:.1f} ms")
    
    benchmark_report = {
        "hardware": "AMD Ryzen 9 8940HX / CPU Execution Provider",
        "input_resolution": f"{cfg['data']['img_size']}x{cfg['data']['img_size']}",
        "file_sizes": {
            "fp32_onnx_mb": fp32_size_mb,
            "int8_onnx_mb": int8_size_mb,
            "target_fp32_mb": cfg["export"]["target_file_size_mb"],
            "target_int8_mb": cfg["export"]["target_quantized_mb"],
            "meets_size_constraint": int8_size_mb < cfg["export"]["target_quantized_mb"]
        },
        "latency_ms": {
            "fp32_mean_ms": mean_fp32,
            "fp32_p95_ms": p95_fp32,
            "int8_mean_ms": mean_int8,
            "int8_p95_ms": p95_int8,
            "target_latency_ms": cfg["export"]["target_cpu_latency_ms"],
            "meets_latency_constraint": mean_int8 < cfg["export"]["target_cpu_latency_ms"]
        },
        "numerical_fidelity": {
            "max_abs_diff_fp32": fp32_diff,
            "max_abs_diff_int8": int8_diff
        }
    }
    
    bench_out = os.path.join(reports_dir, "export_benchmark.json")
    with open(bench_out, "w") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"Benchmark report saved to {bench_out}")

if __name__ == "__main__":
    export_and_quantize()
