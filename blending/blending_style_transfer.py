"""
Blending Style Transfer Module
Combines image blending with neural style transfer using VGG19
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================================
# Utility Functions
# ============================================================================

def ensure_rgb_pil(img: Image.Image | np.ndarray | None) -> Image.Image:
    if img is None:
        raise ValueError("Missing image input.")
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img.convert("RGB")


def ensure_rgba_pil(img: Image.Image | np.ndarray | None) -> Image.Image:
    if img is None:
        raise ValueError("Missing image input.")
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img.convert("RGBA")


def pil_to_rgb01_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> tensor [1,3,H,W], RGB, range [0,1]."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().to(DEVICE)
    return x.clamp(0, 1).contiguous()


def pil_mask_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL L -> tensor [1,1,H,W], range [0,1]."""
    arr = np.asarray(img.convert("L")).astype(np.float32) / 255.0
    arr = arr[..., None]
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().to(DEVICE)
    return x.clamp(0, 1).contiguous()


def rgb01_tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    x = x.permute(1, 2, 0).contiguous().numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return Image.fromarray(x)


def resize_pil_long_side(img: Image.Image, max_side: int, mode: str = "RGB") -> Image.Image:
    """Resize so the longest side <= max_side. Keeps aspect ratio."""
    img = img.convert(mode)
    w, h = img.size
    scale = min(max_side / max(h, w), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if (new_w, new_h) == (w, h):
        return img
    resample = Image.BILINEAR if mode == "RGB" else Image.NEAREST
    return img.resize((new_w, new_h), resample=resample)


def resize_tensor(x: torch.Tensor, size_hw: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if mode in ["bilinear", "bicubic"]:
        return F.interpolate(x, size=size_hw, mode=mode, align_corners=False).contiguous()
    return F.interpolate(x, size=size_hw, mode=mode).contiguous()


# ============================================================================
# Pre/Post-processing for NST (Neural Style Transfer)
# ============================================================================

BGR_MEAN = torch.tensor([0.40760392, 0.45795686, 0.48501961], device=DEVICE).view(1, 3, 1, 1)


def rgb01_to_nst(x_rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB [0,1] to NST space:
    RGB -> BGR -> Normalize(mean=BGR_MEAN, std=1) -> *255
    """
    x_bgr = x_rgb[:, [2, 1, 0], :, :]
    x_bgr = (x_bgr - BGR_MEAN.to(x_bgr.device, x_bgr.dtype)) * 255.0
    return x_bgr.contiguous()


def nst_to_rgb01(x_nst: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """
    Convert NST space back to RGB [0,1]:
    /255 -> add BGR mean -> BGR to RGB -> clip [0,1]
    """
    x_bgr = x_nst / 255.0 + BGR_MEAN.to(x_nst.device, x_nst.dtype)
    x_rgb = x_bgr[:, [2, 1, 0], :, :]
    if clamp:
        x_rgb = x_rgb.clamp(0.0, 1.0)
    return x_rgb.contiguous()


def nst_tensor_to_pil(x_nst: torch.Tensor) -> Image.Image:
    return rgb01_tensor_to_pil(nst_to_rgb01(x_nst, clamp=True))


# ============================================================================
# VGG Model
# ============================================================================

class VGG(nn.Module):
    def __init__(self, pool: str = "max"):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if pool == "max":
            pool_layer = nn.MaxPool2d
        elif pool == "avg":
            pool_layer = nn.AvgPool2d
        else:
            raise ValueError("pool must be 'max' or 'avg'")

        self.pool1 = pool_layer(kernel_size=2, stride=2)
        self.pool2 = pool_layer(kernel_size=2, stride=2)
        self.pool3 = pool_layer(kernel_size=2, stride=2)
        self.pool4 = pool_layer(kernel_size=2, stride=2)
        self.pool5 = pool_layer(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, out_keys: list[str]):
        out = {}
        out["r11"] = F.relu(self.conv1_1(x))
        out["r12"] = F.relu(self.conv1_2(out["r11"]))
        out["p1"] = self.pool1(out["r12"])
        out["r21"] = F.relu(self.conv2_1(out["p1"]))
        out["r22"] = F.relu(self.conv2_2(out["r21"]))
        out["p2"] = self.pool2(out["r22"])
        out["r31"] = F.relu(self.conv3_1(out["p2"]))
        out["r32"] = F.relu(self.conv3_2(out["r31"]))
        out["r33"] = F.relu(self.conv3_3(out["r32"]))
        out["r34"] = F.relu(self.conv3_4(out["r33"]))
        out["p3"] = self.pool3(out["r34"])
        out["r41"] = F.relu(self.conv4_1(out["p3"]))
        out["r42"] = F.relu(self.conv4_2(out["r41"]))
        out["r43"] = F.relu(self.conv4_3(out["r42"]))
        out["r44"] = F.relu(self.conv4_4(out["r43"]))
        out["p4"] = self.pool4(out["r44"])
        out["r51"] = F.relu(self.conv5_1(out["p4"]))
        out["r52"] = F.relu(self.conv5_2(out["r51"]))
        out["r53"] = F.relu(self.conv5_3(out["r52"]))
        out["r54"] = F.relu(self.conv5_4(out["r53"]))
        out["p5"] = self.pool5(out["r54"])
        return [out[key] for key in out_keys]


class GramMatrix(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram.div_(h * w)
        return gram


class GramMSELoss(nn.Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.MSELoss()(GramMatrix()(input), target)


# Style and content layer configurations
STYLE_LAYERS = ["r11", "r21", "r31", "r41", "r51"]
CONTENT_LAYERS = ["r42"]
LOSS_LAYERS = STYLE_LAYERS + CONTENT_LAYERS
DEFAULT_STYLE_LAYER_WEIGHTS = [1e3 / n**2 for n in [64, 128, 256, 512, 512]]
DEFAULT_CONTENT_WEIGHT = 1.0

# VGG model cache
VGG_CACHE = {"path": None, "model": None}


def get_vgg_model(model_path: str | Path = "Models/vgg_conv.pth") -> VGG:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find VGG weight file: {model_path}")

    cache_key = str(model_path.resolve())
    if VGG_CACHE["model"] is not None and VGG_CACHE["path"] == cache_key:
        return VGG_CACHE["model"]

    vgg = VGG(pool="max")
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    vgg.load_state_dict(state)
    vgg.to(DEVICE).eval()
    for param in vgg.parameters():
        param.requires_grad_(False)

    VGG_CACHE["path"] = cache_key
    VGG_CACHE["model"] = vgg
    return vgg


def download_vgg_model():
    """Download VGG model weights if not present."""
    from pathlib import Path
    import subprocess
    import sys

    VGG_MODEL_PATH = Path("Models/vgg_conv.pth")
    VGG_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not VGG_MODEL_PATH.exists():
        try:
            import gdown
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

        url = "https://drive.google.com/uc?id=1lLSi8BXd_9EtudRbIwxvmTQ3Ms-Qh6C8"
        gdown.download(url, str(VGG_MODEL_PATH), quiet=False)

    print(f"VGG model path: {VGG_MODEL_PATH.resolve()}")
    print(f"Exists: {VGG_MODEL_PATH.exists()}")
    return VGG_MODEL_PATH
# ============================================================================
# Blending Loss Functions
# ============================================================================

def laplacian_filter(x: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    c = x.shape[1]
    kernel = kernel.repeat(c, 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=c)


def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(dtype=a.dtype, device=a.device)
    diff = (a - b).pow(2) * mask
    denom = mask.sum() * a.shape[1]
    return diff.sum() / (denom + eps)


def make_boundary(mask: torch.Tensor, radius: int = 5) -> torch.Tensor:
    k = 2 * radius + 1
    dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=radius)
    return (dilated - eroded).clamp(0, 1)


def gradient_laplacian_loss(gen_rgb01: torch.Tensor, source_canvas_rgb01: torch.Tensor, 
                           background_rgb01: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Laplacian loss in RGB [0,1] space for stable blending scale."""
    lap_gen = laplacian_filter(gen_rgb01)
    lap_src = laplacian_filter(source_canvas_rgb01)
    lap_bg = laplacian_filter(background_rgb01)
    inside = masked_mse(lap_gen, lap_src, mask)
    outside = masked_mse(lap_gen, lap_bg, 1.0 - mask)
    return inside + outside


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    loss_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    loss_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return loss_h + loss_w


def history_to_plot(history: list[dict[str, float]]):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    steps = [row["step"] for row in history]
    for key in ["total", "style", "content", "gradient", "boundary", "tv"]:
        ax.plot(steps, [row[key] for row in history], label=key)
    ax.set_xlabel("LBFGS iteration")
    ax.set_ylabel("loss")
    ax.set_title("Loss history")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# ============================================================================
# Main Blending Style Transfer Function
# ============================================================================
def run_style_transfer(
        content_img: Image.Image,
        style_img: Image.Image,
        w_content: float=4.0,
        w_style_total: float=1.0,
        max_side: int = 512,
        num_steps: int=1000,
        vgg_model_path: str = "Models/vgg_conv.pth"
):

    content_img = resize_pil_long_side(content_img.convert("RGB"), max_side, mode="RGB")
    style_img = resize_pil_long_side(style_img.convert("RGB"), max_side, mode="RGB").resize(
        content_img.size, Image.BILINEAR
    )

    style_rgb01 = pil_to_rgb01_tensor(style_img)
    content_rgb01 = pil_to_rgb01_tensor(content_img)

    style_nst = rgb01_to_nst(style_rgb01)
    content_nst = rgb01_to_nst(content_rgb01)

    # Load VGG model
    vgg = get_vgg_model(vgg_model_path)
    gram_loss = GramMSELoss().to(DEVICE)
    mse_loss = nn.MSELoss().to(DEVICE)

    style_layer_weights = [1e3 / n**2 for n in [64, 128, 256, 512, 512]]


    # Extract and freeze target features
    with torch.no_grad():
        style_targets = [GramMatrix()(A).detach() for A in vgg(style_nst, STYLE_LAYERS)]
        content_targets = [A.detach() for A in vgg(content_nst, CONTENT_LAYERS)]
        targets = style_targets + content_targets
        targets = [t.to(DEVICE) for t in targets]



    opt_img = (torch.randn(content_nst.size(), device=DEVICE, dtype=content_nst.dtype) * 1e-3).contiguous()
    opt_img.requires_grad_(True)
    optimizer = optim.LBFGS([opt_img])

    n_iter = [0]
    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, LOSS_LAYERS)
        
        # A. Style Loss
        style_losses = [
            style_layer_weights[i] * gram_loss(activation, targets[i])
            for i, activation in enumerate(out[:len(STYLE_LAYERS)])
        ]
        loss_style = torch.stack(style_losses).sum()

        # B. Content Loss
        content_losses = [
            w_content * mse_loss(activation, targets[len(STYLE_LAYERS) + i])
            for i, activation in enumerate(out[len(STYLE_LAYERS):])
        ]
        loss_content = torch.stack(content_losses).sum()

        # Total loss
        loss = (
            w_style_total * loss_style
            + loss_content
        )

        loss.backward()
        if opt_img.grad is not None:
            opt_img.grad = opt_img.grad.contiguous()

        n_iter[0] += 1
        if n_iter[0] % 20 == 0 or n_iter[0] == 1:
            print(f"Step {n_iter[0]:03d}/{num_steps} | Total: {loss.item():.4f} | "
                  f"Style: {loss_style.item():.4f} | Content: {loss_content.item():.4f} | ")
        return loss

    # Run optimization
    while n_iter[0] < num_steps:
        optimizer.step(closure)

    # 5. Convert result back to PIL image
    gen_rgb01 = nst_to_rgb01(opt_img.detach(), clamp=True)
    result_pil = rgb01_tensor_to_pil(gen_rgb01)
    return result_pil




def run_blending_style_transfer(
    source_img: Image.Image,
    mask_img: Image.Image,
    target_img: Image.Image,
    style_img: Image.Image,
    max_side: int = 512,
    num_steps: int = 300,
    vgg_model_path: str = "Models/vgg_conv.pth",
    w_gradient: float = 400.0,
    w_content: float = 4.0,
    w_style_total: float = 1.0,
    w_boundary: float = 0.0,
    w_tv: float = 0.0,
    boundary_radius: int = 5,
):
    """
    Perform Image Blending + Neural Style Transfer for a single source image.
    
    Args:
        source_img: Source image to blend (PIL Image)
        mask_img: Mask defining the region (PIL Image, L mode)
        target_img: Target/background image (PIL Image)
        style_img: Style reference image (PIL Image)
        max_side: Maximum dimension for resizing
        num_steps: Number of LBFGS optimization steps
        vgg_model_path: Path to VGG model weights
        w_gradient: Weight for gradient/Laplacian loss
        w_content: Weight for content loss
        w_style_total: Weight for style loss
        w_boundary: Weight for boundary loss
        w_tv: Weight for total variation loss
        boundary_radius: Radius for boundary creation
        
    Returns:
        PIL Image: Result of blending and style transfer
    """
    # 1. Resize all images to match target size
    target_img = resize_pil_long_side(target_img.convert("RGB"), max_side, mode="RGB")
    style_img = resize_pil_long_side(style_img.convert("RGB"), max_side, mode="RGB").resize(
        target_img.size, Image.BILINEAR
    )
    
    # Resize source and mask to match target dimensions
    src_img = source_img.convert("RGB").resize(target_img.size, Image.BILINEAR)
    mask_canvas_pil = mask_img.convert("L").resize(target_img.size, Image.NEAREST)

    # 2. Convert to tensors [0, 1] on device
    background_rgb01 = pil_to_rgb01_tensor(target_img)
    source_canvas_rgb01 = pil_to_rgb01_tensor(src_img)
    style_rgb01 = pil_to_rgb01_tensor(style_img)
    mask_canvas = (pil_mask_to_tensor(mask_canvas_pil) > 0.5).float()

    # Create naive composite and boundary mask
    composite_rgb01 = source_canvas_rgb01 * mask_canvas + background_rgb01 * (1.0 - mask_canvas)
    boundary_mask = make_boundary(mask_canvas, radius=boundary_radius)

    # 3. Convert to NST color space (BGR * 255 with mean subtraction)
    composite_nst = rgb01_to_nst(composite_rgb01)
    style_nst = rgb01_to_nst(style_rgb01)

    # Load VGG model
    vgg = get_vgg_model(vgg_model_path)
    gram_loss = GramMSELoss().to(DEVICE)
    mse_loss = nn.MSELoss().to(DEVICE)
    
    # Style layer weights
    style_layer_weights = [1e3 / n**2 for n in [64, 128, 256, 512, 512]]

    # Extract and freeze target features
    with torch.no_grad():
        style_targets = [GramMatrix()(A).detach() for A in vgg(style_nst, STYLE_LAYERS)]
        content_targets = [A.detach() for A in vgg(composite_nst, CONTENT_LAYERS)]
        targets = style_targets + content_targets
        targets = [t.to(DEVICE) for t in targets]

    # 4. Initialize from Gaussian noise
    opt_img = (torch.randn(composite_nst.size(), device=DEVICE, dtype=composite_nst.dtype) * 1e-3).contiguous()
    opt_img.requires_grad_(True)
    optimizer = optim.LBFGS([opt_img])

    n_iter = [0]
    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, LOSS_LAYERS)
        
        # A. Style Loss
        style_losses = [
            style_layer_weights[i] * gram_loss(activation, targets[i])
            for i, activation in enumerate(out[:len(STYLE_LAYERS)])
        ]
        loss_style = torch.stack(style_losses).sum()

        # B. Content Loss
        content_losses = [
            w_content * mse_loss(activation, targets[len(STYLE_LAYERS) + i])
            for i, activation in enumerate(out[len(STYLE_LAYERS):])
        ]
        loss_content = torch.stack(content_losses).sum()

        # C. Blending Losses (computed in RGB [0,1] space)
        gen_rgb01 = nst_to_rgb01(opt_img, clamp=False)
        loss_grad = gradient_laplacian_loss(gen_rgb01, source_canvas_rgb01, background_rgb01, mask_canvas)
        loss_boundary = masked_mse(gen_rgb01, composite_rgb01, boundary_mask)
        loss_tv = total_variation_loss(gen_rgb01)

        # Total loss
        loss = (
            w_style_total * loss_style
            + loss_content
            + w_gradient * loss_grad
            + w_boundary * loss_boundary
            + w_tv * loss_tv
        )

        loss.backward()
        if opt_img.grad is not None:
            opt_img.grad = opt_img.grad.contiguous()

        n_iter[0] += 1
        if n_iter[0] % 20 == 0 or n_iter[0] == 1:
            print(f"Step {n_iter[0]:03d}/{num_steps} | Total: {loss.item():.4f} | "
                  f"Style: {loss_style.item():.4f} | Content: {loss_content.item():.4f} | "
                  f"Grad: {loss_grad.item():.4f}")
        return loss

    # Run optimization
    while n_iter[0] < num_steps:
        optimizer.step(closure)

    # 5. Convert result back to PIL image
    gen_rgb01 = nst_to_rgb01(opt_img.detach(), clamp=True)
    result_pil = rgb01_tensor_to_pil(gen_rgb01)
    return result_pil
