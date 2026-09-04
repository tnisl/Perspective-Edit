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

Install `spandrel`:
```bash
pip install spandrel==0.4.1 --no-deps
```

Install the requirements:
```bash
pip install -r requirements.txt
```

