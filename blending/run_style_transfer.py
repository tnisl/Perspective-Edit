
"""
Demo script for running blending style transfer
This script demonstrates how to use the 4 required input variables:
- left_face_mask: Mask image defining the region
- left_face_mask_content: Source content image to blend
- target_image: Background/target image
- style_image: Style reference image
"""

import os
import matplotlib
matplotlib.use("Agg")

from PIL import Image
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter.ip_adapter import IPAdapter
from huggingface_hub import hf_hub_download

import argparse


def run(blend_path, style_path, w_content, w_style, iters):
    
    # Load the 4 required input images from Kaggle datasets
    print("\nLoading input images from Kaggle datasets...")
    
    blend_img = Image.open(blend_path)
    style_img = Image.open(style_path)

    print(f"Blend size: {blend_img.size}")
    print(f"Style size: {style_img.size}")
    
    # Run the blending style transfer
    print("\nStarting blending style transfer...")
    
    base_model_path = "runwayml/stable-diffusion-v1-5"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    ip_ckpt = "ip-adapter_sd15.bin"
    device = "cuda"
    
    # Download IP-Adapter checkpoint if not present
    if not os.path.exists(ip_ckpt):
        print(f"Downloading IP-Adapter checkpoint: {ip_ckpt}")
        ip_ckpt = hf_hub_download(repo_id="h94/IP-Adapter", filename="models/ip-adapter_sd15.bin", repo_type="model")
        print(f"Download complete. File saved to: {ip_ckpt}")
    
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )
    
    # "face id" is not supported yet.
    # Based on IP-Adapter library, constructor is: IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
    ip_model = IPAdapter(pipe, "openai/clip-vit-large-patch14", ip_ckpt, device)
    
    images = ip_model.generate(pil_image=style_img, num_samples=1, image=blend_img, strength=w_style, scale=w_content, num_inference_steps=iters, seed=42)
    
    # Save and display result
    print("\nSaving result...")
    images[0].save('result.png')
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--blend_path", type=str)
    parser.add_argument("--style_path", type=str)
    parser.add_argument("--w_content", type=float, default=8.0)
    parser.add_argument("--w_style", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=300)

    args = parser.parse_args()

    run(args.blend_path, args.style_path, args.w_content, args.w_style, args.iters)




