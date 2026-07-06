import os
import sys
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import subprocess
import numpy as np

import matplotlib.pyplot as plt
import argparse



BASE_DIR = os.getcwd()
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
NUM_VIEWS = 6
HEIGHT, WIDTH = 768, 768
MV_ADAPTER_CODE_DIR = os.path.join(BASE_DIR, "mv_adapter")

sys.path.append(MV_ADAPTER_CODE_DIR)
sys.path.append(os.path.join(MV_ADAPTER_CODE_DIR, "scripts"))

CHECKPOINT_DIR = os.path.join("mv_adapter", "checkpoints")

from inference_ig2mv_sdxl import prepare_pipeline, remove_bg, preprocess_image, run_pipeline
from mvadapter.utils import get_orthogonal_camera, tensor_to_image, make_image_grid, get_plucker_embeds_from_cameras_ortho
from mvadapter.utils.mesh_utils.render import NVDiffRastContextWrapper, render
from mvadapter.utils.mesh_utils.mesh import load_mesh
from texture import TexturePipeline, ModProcessConfig

from huggingface_hub import hf_hub_download



def run_texturing(img_path, mesh_path, output_path):
    ### multivew
    birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
    birefnet.to(DEVICE)
    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    
    
    
    
    remove_bg_fn = lambda x: remove_bg(x, birefnet, lambda img: transform_image(img).half(), DEVICE)
    
    
    
    
    image = Image.open(img_path) 
    
    mv_adapter_pipe = prepare_pipeline(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        unet_model=None,
        lora_model=None,
        adapter_path="huanngzh/mv-adapter",
        scheduler=None,
        num_views=NUM_VIEWS,
        device='cuda:0',
        dtype=torch.float16,
    )
    
    mv_adapter_pipe.enable_vae_tiling()
    
    
    mv_adapter_pipe.unet.to("cuda:1")
    
    if hasattr(mv_adapter_pipe, "image_adapter") and mv_adapter_pipe.image_adapter is not None:
        mv_adapter_pipe.image_adapter.to("cuda:1")
    if hasattr(mv_adapter_pipe, "cond_encoder") and mv_adapter_pipe.cond_encoder is not None:
        mv_adapter_pipe.cond_encoder.to("cuda:1")
    
    mv_adapter_pipe.vae.enable_tiling()
    
    type(mv_adapter_pipe)._execution_device = property(lambda self: torch.device("cuda:0"))
    
    
    mv_images, pos_images, normal_images, ref_image = run_pipeline(
            mv_adapter_pipe,
            mesh_path = mesh_path,
            num_views=NUM_VIEWS,
            text="high quality",
            image=image,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=50,
            guidance_scale=3.0,
            seed=42,
            remove_bg_fn=remove_bg_fn,
            reference_conditioning_scale=1.0,
            negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
            device=DEVICE
        )
    
    
    grid_img = make_image_grid(mv_images, rows=1).save('portrait_views.png')
    
    
    ### texturing
    print("Create texture pipeline")
    texture_pipe = TexturePipeline(
        upscaler_ckpt_path="mv_adapter/checkpoints/RealESRGAN_x2plus.pth",
        inpaint_ckpt_path="mv_adapter/checkpoints/big-lama.pt",
        device=DEVICE
    )
    
    base_az = [0, 90, 180, 270, 180, 180]
    
    corrected_az = [(az - 90) % 360 for az in base_az] 
    
    
    print("running pipeline")
    textured_glb_path = texture_pipe(
        mesh_path = mesh_path,
        save_dir=output_path,
        save_name="result",
        uv_unwarp=True,
        uv_size=4096,
        rgb_path='portrait_views.png',     
        rgb_process_config=ModProcessConfig(view_upscale=False, inpaint_mode="view"), 
        camera_azimuth_deg=corrected_az, 
    )
    
    print(f"Done, saved at: {textured_glb_path}")

if __name__ == "__main__":


    hf_hub_download(
        repo_id='dtarnow/UPscaler', 
        filename='RealESRGAN_x2plus.pth', 
        local_dir='./mv_adapter/checkpoints')
    
    subprocess.run(["wget",
                    "-q",
                    "--show-progress",
                    "-O", CHECKPOINT_DIR + "/" + "big-lama.pt",
                    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"],
                   check=True)


    parser = argparse.ArgumentParser()

    parser.add_argument("--img_path", type=str, help="Image path")
    parser.add_argument("--mesh_path", type=str, help="Mesh object path")
    parser.add_argument("--output_path", type=str, default=os.getcwd(), help="Output path")

    args = parser.parse_args()

    run_texturing(args.img_path, args.mesh_path, args.output_path)
    
