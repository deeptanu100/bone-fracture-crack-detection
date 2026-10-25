"""
Lightweight Model Architecture for Bone Fracture & Crack Detection with Built-in Grad-CAM.
Designed to meet strict local constraints: <25 MB FP32 (<10 MB INT8), <300ms CPU latency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FractureDetector(nn.Module):
    """
    MobileNetV3-Large based classifier with progressive feature maps
    and direct Grad-CAM activation hooking.
    """
    def __init__(self, architecture="mobilenet_v3_large", pretrained=True, dropout=0.2):
        super().__init__()
        self.architecture = architecture
        
        if architecture == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_large(weights=weights)
            self.features = base.features
            in_features = 960
            self.target_layer = self.features[-1]
        elif architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            self.features = base.features
            in_features = 1280
            self.target_layer = self.features[-1]
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
            
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1)
        )
        
        # Hooks storage for Grad-CAM
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            
        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])
        
    def forward(self, x):
        feat = self.features(x)
        pooled = self.pool(feat)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits.squeeze(-1)

    def generate_gradcam(self, x, target_class=None):
        """
        Compute pixel-level Grad-CAM heatmap over the input image.
        Returns normalized [0, 1] numpy heatmap of shape (H, W).
        """
        self.eval()
        logits = self(x)
        probs = torch.sigmoid(logits)
        
        self.zero_grad()
        logits.backward(retain_graph=True)
        
        # Global average pool the gradients
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        
        # Weight the channels by the pooled gradients
        for i in range(len(pooled_grads)):
            activations[i, :, :] *= pooled_grads[i]
            
        heatmap = torch.mean(activations, dim=0).detach().cpu()
        heatmap = F.relu(heatmap)
        
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.numpy(), probs.item()
