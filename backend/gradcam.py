import torch
import numpy as np
import cv2

class GradCAM:
    """
    Grad-CAM for nn.Module model built with ResNet backbone at model.img
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Locate last conv layer in resnet backbone (layer4[-1].conv2)
        # This assumes model.img is a torchvision.resnet model.
        try:
            self.target_layer = model.img.layer4[-1].conv2
        except Exception as e:
            # fallback: try to find last Conv2d by iterating modules
            self.target_layer = None
            for m in reversed(list(model.modules())):
                if isinstance(m, torch.nn.Conv2d):
                    self.target_layer = m
                    break
            if self.target_layer is None:
                raise RuntimeError("Could not find a conv layer in model for GradCAM.") from e

        # register hooks
        self.handle_fwd = self.target_layer.register_forward_hook(self._fwd_hook)
        self.handle_bwd = self.target_layer.register_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, input, output):
        self.activations = output

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, input_tensor):
        # input_tensor: (1,C,H,W) on correct device
        self.model.zero_grad()
        out = self.model.img(input_tensor)  # forward through backbone only
        # choose scalar target as mean of output features (works as proxy)
        target = out.mean()
        target.backward(retain_graph=True)

        grads = self.gradients  # (B,C,H,W)
        acts = self.activations  # (B,C,H,W)
        weights = torch.mean(grads, dim=(2,3), keepdim=True)  # (B,C,1,1)
        gcam = torch.sum(weights * acts, dim=1)  # (B,H,W)
        gcam = torch.relu(gcam)
        gcam = gcam.squeeze().cpu().detach().numpy()
        # normalize 0..1
        gcam -= gcam.min()
        if gcam.max() != 0:
            gcam /= gcam.max()
        return gcam

def overlay_heatmap(original_rgb, cam):
    h, w, _ = original_rgb.shape
    cam_res = cv2.resize(cam, (w, h))
    cam_uint8 = np.uint8(255 * cam_res)
    heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_rgb, 0.6, heat, 0.4, 0)
    return overlay

