"""
Demo script for facial landmark detection and face blending
Matches the exact flow of facial-landmark.ipynb notebook
Demonstrates the 4 required input variables:
- IMG_PATH: Original portrait image
- ROTATED_IMG_PATH: Rotated/transformed portrait image
- MASK_PATH: Binary mask for face region
- BISENET_WEIGHT_PATH: BiSeNet model weights (used in segmentation)

Outputs only the final blended result.
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_alignment
import torch
import torchvision.transforms as transforms
import argparse
import json

# Disable torch.compile for compatibility
torch.compile = lambda model, *args, **kwargs: model

from model import BiSeNet

HEIGHT, WIDTH = 512, 512

BISENET_WEIGHT_PATH = 'facial_landmark/79999_iter.pth'

def facial_landmark(img, fa):
    """Detect facial landmarks"""
    h, w, _ = img.shape
    bbox = [0, 0, w, h]
    preds = fa.get_landmarks_from_image(img, detected_faces=[bbox])
    return preds


def load_bisenet_model(weight_path='79999_iter.pth'):
    """Load BiSeNet face segmentation model"""
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    net.to(device)
    net.eval()
    return net, device



def run(img_path, rotated_img_path, mask_path):
    print("Loading images and detecting landmarks...")
    
    # Initialize face alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')
    
    # Load images
    img = cv2.imread(img_path)
    img = cv2.resize(img, (WIDTH, HEIGHT))

    
    rotated_img = cv2.imread(rotated_img_path)
    rotated_img = cv2.resize(rotated_img, (WIDTH, HEIGHT))

    
    
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (WIDTH, HEIGHT))

    
    
    # Detect landmarks
    preds = facial_landmark(img, fa)
    rotated_preds = facial_landmark(rotated_img, fa)
    
    print("Calculating centroids...")
    
    # Calculate centroid for original image
    bucket = []
    for i, pts in enumerate(preds[0]):
        x, y = int(pts[0]), int(pts[1])
        if y < mask.shape[0] and x < mask.shape[1] and mask[y, x, 0] != 0:
            bucket.append([x, y])
    
    bucket = np.array(bucket)
    centroid = np.mean(bucket, axis=0)
    
    # Calculate centroid for rotated image
    bucket = []
    for i, pts in enumerate(preds[0]):
        x, y = int(pts[0]), int(pts[1])
        if y < mask.shape[0] and x < mask.shape[1] and mask[y, x, 0] != 0:
            dx, dy = int(rotated_preds[0][i][0]), int(rotated_preds[0][i][1])
            bucket.append([dx, dy])
    
    bucket = np.array(bucket)
    rotated_centroid = np.mean(bucket, axis=0)
    
    print("Performing translation and warping...")
    
    # Calculate translation
    vector = rotated_centroid - centroid
    translation_matrix = np.float32([[1, 0, vector[0]],
                                     [0, 1, vector[1]]])
    
    reverse_matrix = np.float32([[1, 0, -vector[0]],
                                 [0, 1, -vector[1]]])
    
    moved_mask = cv2.warpAffine(mask, translation_matrix, (WIDTH, HEIGHT), borderValue=(0, 0, 0))

    moved_mask_float = moved_mask.astype(np.float32) / 255.0

    
    # Calculate bounding box for seamless clone
    
    mask_gray = cv2.cvtColor(moved_mask, cv2.COLOR_BGR2GRAY)

    mask_gray = mask_gray.astype(np.uint8)
    
    x, y, w, h = cv2.boundingRect(mask_gray)
    center = (x + w // 2, y + h // 2)

    moved_center = (center[0] - vector[0], center[1] - vector[1])

    
    
    # Final seamless clone with segmented face
    result = cv2.seamlessClone(
        rotated_img,
        img,
        moved_mask,
        (int(moved_center[0]), int(moved_center[1])),
        cv2.NORMAL_CLONE
    )

    cv2.imwrite('materials/poisson_blending_result.png', result)
    content_mask = (rotated_img * moved_mask_float).astype(np.uint8)

    moved_content_mask = cv2.warpAffine(content_mask, reverse_matrix, (WIDTH, HEIGHT), borderValue=(0, 0, 0))

    cv2.imwrite('materials/moved_content_mask.png', moved_content_mask)

    data = {
        "translation_matrix": translation_matrix.tolist(),
        "reverse_matrix": reverse_matrix.tolist()
    }
    with open("materials/vector.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--img_path", type=str)
    parser.add_argument("--rotated_img_path", type=str)
    parser.add_argument("--mask_path", type=str)

    args = parser.parse_args()


    run(args.img_path, args.rotated_img_path, args.mask_path)

    
