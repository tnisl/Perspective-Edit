
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
from blending_style_transfer import run_style_transfer, download_vgg_model

import argparse


def run(blend_path, style_path, w_content, w_style, iters):
    # Download VGG model if not present
    print("Checking VGG model...")
    download_vgg_model()
    
    # Load the 4 required input images from Kaggle datasets
    print("\nLoading input images from Kaggle datasets...")
    
    blend_img = Image.open(blend_path)
    style_img = Image.open(style_path)

    print(f"Blend size: {blend_img.size}")
    print(f"Style size: {style_img.size}")
    
    # Run the blending style transfer
    print("\nStarting blending style transfer...")
    result = run_style_transfer(
        content_img=blend_img,
        style_img=style_img,
        w_content=w_content,
        w_style_total=w_style,
        num_steps=iters,  # Adjust for quality vs speed tradeoff
        max_side=512
    )
    
    # Save and display result
    print("\nSaving result...")
    result.save('result.png')
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--blend_path", type=str)
    parser.add_argument("--style_path", type=str)
    parser.add_argument("--w_content", type=float, default=8.0)
    parser.add_argument("--w_style", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=300)

    args = parser.parse_args()

    run(args.blend_path, args.style_path, args.w_content, args.w_style, args.iters)




