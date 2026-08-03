import gradio as gr
import cv2
import numpy as np
import subprocess

def extract_materials(editor_data):
    if editor_data is None:
        return None
        
    img_rgb = editor_data["background"] 
    
    layers = editor_data["layers"]
    
    if not layers:
        mask_1ch = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    else:
        brush_layer = layers[0]
        
        alpha_channel = brush_layer[:, :, 3]
        
        mask_1ch = (alpha_channel > 0).astype(np.uint8) * 255
        

    cv2.imwrite('materials/mask.png', mask_1ch)
    cv2.imwrite('materials/portrait.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def export_3d(editor_data):
    extract_materials(editor_data)

    subprocess.run([".venv/bin/python",
                    "3d_scripts/inference_triposg.py",
                    "--image-input", "materials/portrait.png",
                    "--output-path", "./materials/object.glb"])
    subprocess.run([".venv/bin/python",
                   "run_texturing.py",
                   "--img_path", "materials/portrait.png",
                   "--mesh_path", "materials/object.glb",
                   "--output_path", "materials"])

    return "materials/mvadapter_model_result.glb"




def run(editor_data, style_img,  azimuth, elevation, iters):
    extract_materials(editor_data)
    cv2.imwrite('materials/style_img.png', cv2.cvtColor(style_img, cv2.COLOR_RGB2BGR))
    subprocess.run([".venv/bin/python",
                   "run_rotation.py",
                   "--mesh_path", "materials/mvadapter_model_result.glb",
                   "--output_path", "materials/rotated_img.png",
                   "--azimuth", f"{azimuth}",
                    "--elevation", f"{elevation}"])
    subprocess.run([".venv/bin/python", "facial_landmark/run_facial_landmark.py",
                    "--img_path", "materials/portrait.png",
                    "--rotated_img_path", "materials/rotated_img.png",
                    "--mask_path", "materials/mask.png"])
   # blending + style transfer
   # subprocess.run([".venv/bin/python", "blending/run_blending_style_transfer.py",
   #                 "--source_path", "materials/moved_content_mask.png",
   #                 "--target_path", "materials/portrait.png", 
   #                 "--mask_path", "materials/mask.png",
   #                 "--style_path", "materials/style_img.png", 
   #                 "--iters", f"{iters}"])

    # style transfer only
    subprocess.run([".venv/bin/python", "blending/run_style_transfer.py",
                    "--blend_path", "materials/poisson_blending_result.png",
                    "--style_path", "materials/style_img.png", 
                    "--iters", f"{iters}"])

    result=cv2.imread('result.png')

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

with gr.Blocks() as demo:
    gr.Markdown("Perspective Edit")
    
    with gr.Row():
        with gr.Column():

            with gr.Row():
                input_editor = gr.ImageEditor(
                    label="Upload portrait and draw mask:",
                    type="numpy",
                    sources=["upload"]
                )

                style_img = gr.Image(
                    label="Style image:",
                    type="numpy",
                    sources=["upload", "clipboard"]
                )
            btn_export3d = gr.Button("Export 3D", variant="primary")
            btn_submit = gr.Button("Run", variant="primary")

            with gr.Column():
                azimuth=gr.Slider(
                    minimum=-180,
                    maximum=180,
                    value=0,
                    step=1,
                    label="Azimuth"
                )

                elevation=gr.Slider(
                    minimum=-180,
                    maximum=180,
                    value=0,
                    step=1,
                    label="Elevation"
                )
                iters=gr.Slider(
                    minimum=0,
                    maximum=10000,
                    value=2000,
                    step=1,
                    label="Iterations"
                )

            
        with gr.Column():
            object_3d = gr.Model3D(label="3D portrait", clear_color=[0.8, 0.8, 0.8, 1.0])
            output = gr.Image(label="Result")
            
    btn_export3d.click(
        fn=export_3d,
        inputs=[input_editor],
        outputs=[object_3d]
    )
    btn_submit.click(
        fn=run, 
        inputs=[input_editor, style_img, azimuth, elevation, iters], 
        outputs=[output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
