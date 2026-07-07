import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import argparse


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.getcwd()
MV_ADAPTER_CODE_DIR = os.path.join(BASE_DIR, "mv_adapter")
if MV_ADAPTER_CODE_DIR not in sys.path:
    sys.path.append(MV_ADAPTER_CODE_DIR)


SCALE = 0.8

from mvadapter.utils.mesh_utils.render import NVDiffRastContextWrapper, render
from mvadapter.utils.mesh_utils.mesh import load_mesh
from mvadapter.utils.mesh_utils.camera import get_orthogonal_camera


def rotate(mesh_path, output_path, angle):
    print("="*10)
    print("Rendering new view")

    print("Initialize context (render)")
    ctx = NVDiffRastContextWrapper(device = DEVICE, context_type = DEVICE)
    
    print("Loading mesh")
    mesh = load_mesh(mesh_path, rescale = True, device = DEVICE)


    print("Create camera")
    camera = get_orthogonal_camera(
        elevation_deg=[0], azimuth_deg=[angle - 90],
        distance=[1.8], left=-0.55*SCALE, right=0.55*SCALE, bottom=-0.55*SCALE, top=0.55*SCALE, device=DEVICE
    )

    
    print("Rendering")
    
    render_out = render(ctx, mesh, camera, height=768, width=768, render_attr=True)
    
    color_tensor = None
    
    for attr in ['attr', 'color', 'rgb', 'image']:
        if hasattr(render_out, attr):
            val = getattr(render_out, attr)
    
            if val is not None and val.dim()==4:
                color_tensor = val
                break
    
    img_3d = (np.clip(color_tensor[0].cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)
    cv2.imwrite(output_path, img_3d)
    print("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mesh_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--angle", type=float)

    args = parser.parse_args()

    rotate(args.mesh_path, args.output_path, args.angle)






