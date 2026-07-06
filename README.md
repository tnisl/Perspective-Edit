# Persepctive-Edit
### Install requirements

Build `torch` compiled with your gpu:
```bash
pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cu{your_cuda_version}
```

Install `diso` and `NVDiffRast`:
```bash
pip install diso --no-build-isolation
pip install -p .venv git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

Install 'spandrel':
```bash
pip install spandrel==0.4.1 --no-deps
```

Install the requirements:
```bash
pip install -r requirements.txt
```

Next, install some checkpoints for the mv_adapter:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='dtarnow/UPscaler', 
    filename='RealESRGAN_x2plus.pth', 
    local_dir='./Perspective-Edit/mv_adapter/checkpoints')
```

```bash
wget -q --show-progress -O /kaggle/working/Perspective-Edit/mv_adapter/checkpoints/big-lama.pt https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt
```


