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

# Disable torch.compile for compatibility
torch.compile = lambda model, *args, **kwargs: model

from model import BiSeNet



BISENET_WEIGHT_PATH = '79999_iter.pth'

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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    rotated_img = cv2.imread(rotated_img_path)
    rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
    rotated_img = cv2.resize(rotated_img, (img.shape[0], img.shape[1]))
    
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (1024, 1024))
    
    # Detect landmarks
    preds = facial_landmark(img, fa)
    rotated_preds = facial_landmark(rotated_img, fa)
    
    print("Calculating centroids...")
    
    # Calculate centroid for original image
    bucket = []
    for i, pts in enumerate(preds[0]):
        x, y = int(pts[0]), int(pts[1])
        if y <= mask.shape[0] and x <= mask.shape[1] and mask[y, x, 0] != 0:
            bucket.append([x, y])
    
    bucket = np.array(bucket)
    centroid = np.mean(bucket, axis=0)
    
    # Calculate centroid for rotated image
    bucket = []
    for i, pts in enumerate(preds[0]):
        x, y = int(pts[0]), int(pts[1])
        if y <= mask.shape[0] and x <= mask.shape[1] and mask[y, x, 0] != 0:
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
    
    # Calculate bounding box for seamless clone
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()
    
    if mask_gray.max() <= 1.0:
        mask_gray = (mask_gray * 255).astype(np.uint8)
    else:
        mask_gray = mask_gray.astype(np.uint8)
    
    x, y, w, h = cv2.boundingRect(mask_gray)
    c_bbox = (x + w // 2, y + h // 2)
    
    print("Loading BiSeNet model for face segmentation...")
    
    # Load BiSeNet model
    net, device = load_bisenet_model(BISENET_WEIGHT_PATH)
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    print("Segmenting face...")
    
    # Read and prepare image for BiSeNet
    img_bgr = cv2.imread(rotated_img_path)
    img_bgr = cv2.resize(img_bgr, (1024, 1024))
    h_orig, w_orig = img_bgr.shape[:2]
    
    img_resized = cv2.resize(img_bgr, (512, 512))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor_img = transform(img_rgb).unsqueeze(0)
    
    # Run BiSeNet inference
    tensor_img = tensor_img.to(device)
    
    with torch.no_grad():
        output = net(tensor_img)[0]
        parsing_mask = output.squeeze(0).argmax(0).cpu().numpy()
    
    # Extract face labels
    face_ids = [1, 2, 3, 4, 5, 10, 11, 12, 13]
    pure_face_mask = np.isin(parsing_mask, face_ids).astype(np.uint8) * 255
    pure_face_mask_orig = cv2.resize(pure_face_mask, (w_orig, h_orig))
    
    mask_3ch = np.repeat(pure_face_mask_orig[:, :, np.newaxis], 3, axis=2) / 255.0
    only_face_result = (img_bgr * mask_3ch).astype(np.uint8)
    
    print("Creating final seamless blend...")
    
    # Final seamless clone with segmented face
    result = cv2.seamlessClone(
        cv2.cvtColor(only_face_result, cv2.COLOR_BGR2RGB),
        img,
        pure_face_mask_orig,
        (int(c_bbox[0]), int(c_bbox[1])),
        cv2.NORMAL_CLONE
    )
    
    # Save final result
    plt.figure(figsize=(10, 10))
    plt.imshow(result)
    plt.axis('off')
    plt.title('Final Result: Seamless Face Blending with Segmentation')
    plt.tight_layout()
    plt.savefig('final_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print("Processing completed!")
    print("="*60)
    print("\nOutput: final_result.png")
    print(f"\nCentroid (original): {centroid}")
    print(f"Centroid (rotated): {rotated_centroid}")
    print(f"Bounding box center: {c_bbox}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--img_path", type=str)
    parser.add_argument("--rotated_img_path", type=str)
    parser.add_argument("--mask_path", type=str)

    args = parser.parse_args()


    run(args.img_path, args.rotated_img_path, args.mask_path)

    
