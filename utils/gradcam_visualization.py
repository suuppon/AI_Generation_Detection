import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import Dict, List, Optional


class GradCAM:
    """GradCAM 구현 클래스"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor: Dict[str, torch.Tensor], target_class: Optional[int]) -> np.ndarray:
        self.model.zero_grad()
        outputs = self.model(input_tensor['texture'], input_tensor['edge'], input_tensor['other'])
        
        if target_class is None:
            target_class = outputs['combined'].argmax(dim=1)
        loss = outputs['combined'][0, target_class]
        
        loss.backward()
        
        gradients = self.gradients
        activations = self.activations
        if gradients is None or activations is None:
            raise ValueError("Gradients or activations not captured. Check target layer.")
        
        weights = torch.mean(gradients, dim=[2, 3])
        
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy()
