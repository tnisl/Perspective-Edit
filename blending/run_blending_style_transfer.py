"""
Demo script for running blending style transfer
This script demonstrates how to use the 4 required input variables:
- left_face_mask: Mask image defining the region
- left_face_mask_content: Source content image to blend
- target_image: Background/target image
- style_image: Style reference image
"""

import matplotlib
matplotlib.use("Agg")

from PIL import Image
import matplotlib.pyplot as plt
from blending_style_transfer import run_blending_style_transfer, download_vgg_model

import argparse


def run(source_path, target_path, mask_path, style_path, iters):
    # Download VGG model if not present
    print("Checking VGG model...")
    download_vgg_model()
    
    # Load the 4 required input images from Kaggle datasets
    print("\nLoading input images from Kaggle datasets...")
    
    source_img = Image.open(source_path)
    target_img = Image.open(target_path)
    mask = Image.open(mask_path)
    style_img = Image.open(style_path)

    print(f"Mask size: {mask.size}")
    print(f"Source size: {source_img.size}")
    print(f"Target size: {target_img.size}")
    print(f"Style size: {style_img.size}")
    
    # Run the blending style transfer
    print("\nStarting blending style transfer...")
    result = run_blending_style_transfer(
        source_img=source_img,
        mask_img=mask,
        target_img=target_img,
        style_img=style_img,
        num_steps=iters,  # Adjust for quality vs speed tradeoff
        max_side=512
    )
    
    # Save and display result
    print("\nSaving result...")
    result.save('result.png')
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_path", type=str)
    parser.add_argument("--target_path", type=str)
    parser.add_argument("--mask_path", type=str)
    parser.add_argument("--style_path", type=str)
    parser.add_argument("--iters", type=int, default=300)

    args = parser.parse_args()

    run(args.source_path, args.target_path, args.mask_path, args.style_path, args.iters)




