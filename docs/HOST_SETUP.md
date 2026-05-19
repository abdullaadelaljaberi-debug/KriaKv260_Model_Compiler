# Host PC setup

One-time setup of the development PC where you'll quantize and compile
PyTorch checkpoints into `.xmodel` files. Once this is done, you can
recompile any model with `bash scripts/host/02_compile.sh` without
touching this document again.

## Hardware requirements

- **CPU**: x86_64 (the Vitis-AI Docker image is x86_64-only — does not run on ARM hosts)
- **GPU**: NVIDIA with ≥8 GB VRAM, CUDA 11.x compatible (RTX 20-series and newer)
- **RAM**: 16 GB minimum, 32 GB recommended
- **Disk**: 50 GB free (Vitis-AI image ~25 GB unpacked, plus quantization intermediates)
- **Network**: stable broadband (10 GB+ initial download)

## Software requirements

- Ubuntu 22.04 LTS or 24.04 LTS
- Docker 20.10+
- NVIDIA driver 525+
- nvidia-container-toolkit
- Git, Python 3.10+

Pipeline tested on these versions; older may work but unsupported.

## Step-by-step setup

### 0. Verify your starting point

```bash
git clone https://github.com/abdullaadelaljaberi-debug/KriaKv260_Model_Compiler.git
cd KriaKv260_Model_Compiler
bash scripts/host/00_check_prereqs.sh
```

This script tells you exactly what's missing. If everything passes, jump to
step 4. Otherwise, the script's output names the install commands you need.

### 1. Install Docker

```bash
# Quick install (from Docker's official script):
curl -fsSL https://get.docker.com | sudo sh

# Add yourself to the docker group so you don't need sudo every time:
sudo usermod -aG docker $USER

# Apply the group change without logging out:
newgrp docker

# Verify:
docker run --rm hello-world
```

If `hello-world` doesn't work, log out and back in.

### 2. Install NVIDIA driver

If `nvidia-smi` already works, skip to step 3.

```bash
# Check what's available:
ubuntu-drivers devices

# Install the recommended driver (typically 535 or 550):
sudo apt update
sudo apt install -y nvidia-driver-535

# Reboot:
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
```

You should see your GPU model, driver version (≥520), and a list of GPU
processes (probably empty).

### 3. Install NVIDIA Container Toolkit

Lets Docker containers see your GPU.

```bash
# Add the repo:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install + configure:
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify:
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

The verification command should show the same `nvidia-smi` output as on the
host. If it doesn't, the toolkit isn't wired up correctly — check
`docker info | grep -i runtime`.

### 4. Pull the Vitis-AI 3.5 image

```bash
bash scripts/host/01_install_vai.sh
```

This is a ~10 GB download. The script verifies the image works after
pulling. Re-running it is safe — it skips the pull if already done.

#### Optional: Vitis-AI ONNX image (for ONNX-based PTQ work)

The default `vitis-ai-pytorch-cpu` image handles NNDCT (PyTorch-native)
PTQ. For experimenting with the alternative ONNX-based PTQ via
`vai_q_onnx`, pull the ONNX variant:

```bash
docker pull xilinx/vitis-ai-onnx-cpu:latest
```

This image is ~25 GB. **Not required** for the default compile pipeline
(`scripts/host/02_compile.sh` uses NNDCT). Pull only if you're
specifically investigating the ONNX path; see
[YOLOV11.md "ONNX deployment path"](./YOLOV11.md#onnx-deployment-path-investigated-not-deployable)
for the v0.10 outcome on that path.

If you do work with the ONNX image and need additional Python packages
(`onnx-simplifier`, `onnxruntime` with specific provider versions,
etc.), build a derived image rather than pip-installing into the
container at runtime. See
[TROUBLESHOOTING.md → Pip install fails as non-root](./TROUBLESHOOTING.md#pip-install-fails-as-non-root-inside-the-vitis-ai-conda-env)
for the rationale and a sample `Dockerfile`.

### 5. Verify everything

```bash
bash scripts/host/00_check_prereqs.sh
```

All ten checks should now show ✓.

## Compile your first model

You need:

1. A trained `.pt` checkpoint from one of the fully-supported families:
   Ultralytics YOLOv5, Ultralytics YOLOv11 (requires DPU-friendly training —
   see [YOLOV11.md](./YOLOV11.md)), or Megvii YOLOX.
2. A folder of representative calibration images (≥100, ideally 200+)

Place them in the repo:

```
data/weights/yolov5n_lpr.pt
data/calib/<your-calibration-images>.{jpg,png,bmp}
```

Run the compile:

```bash
bash scripts/host/02_compile.sh yolov5 yolov5n \
     data/weights/yolov5n_lpr.pt \
     data/calib/
```

About 5-10 minutes on an RTX 3060 with 200 calibration images. Output:

```
out/yolov5n/yolov5n_kv260.xmodel
```

## Where to get model weights

The pipeline expects user-provided checkpoints. We don't ship weights
with the repo because:

- They're large (10-50 MB each)
- License terms differ per model family (YOLOv5 = AGPL, YOLOX = Apache 2.0,
  etc.) — you should keep awareness of which license applies to your weights
- Checkpoints are use-case-specific (license plate detection vs cars vs faces)

Sources for weights, in order of effort:

1. **Pretrained from upstream** (no training needed — coarse, generic):
   - YOLOv5: https://github.com/ultralytics/yolov5/releases
   - YOLOX: https://github.com/Megvii-BaseDetection/YOLOX/releases
2. **Fine-tuned from upstream** (some training needed — best for thesis):
   Train upstream on your dataset for your specific use case (license
   plates, parts inspection, etc.). Each upstream repo has a clear training
   guide. This is what the LPR demo uses.
3. **From the AMD model zoo** (already trained, sometimes already quantized):
   https://github.com/Xilinx/Vitis-AI/tree/v3.5/model_zoo — these are
   pretrained on standard benchmarks (COCO, Cityscapes, etc.). Useful for
   benchmarking but not for application-specific use.

## Calibration image guidance

The quantizer uses calibration images to learn activation ranges. Quality
matters more than quantity:

- **Coverage**: include the full diversity the deployed model will see —
  different lighting, angles, distances, classes
- **Quantity**: 100 minimum, 200 typical, 500 if you can spare it
- **Source**: ideally a held-out subset of your training data
- **Format**: JPG/PNG/BMP — anything PIL can open

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `docker: permission denied` | Not in docker group | `sudo usermod -aG docker $USER && newgrp docker` |
| `nvidia-smi: command not found` | Driver not installed | Install per step 2 above |
| `could not select device driver "" with capabilities: [[gpu]]` | Container toolkit missing | Install per step 3 above |
| `bash: vai_c_xir: command not found` | Running outside the Docker container | Always use `scripts/host/02_compile.sh` to run compiles |
| Vitis-AI pull stalls at ~9 GB | Slow internet or Docker Hub rate limit | `docker login` (free account) and retry |
| Out of disk space mid-compile | Build dir grew large | Run `docker system prune -a` and check disk |
