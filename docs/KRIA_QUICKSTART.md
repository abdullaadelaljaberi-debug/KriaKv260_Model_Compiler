# Kria KV260 Quickstart — From Box to Running xmodel

A condensed reference for getting a Kria KV260 from out-of-the-box to
running a custom-compiled `.xmodel` on the DPU. Every command block is
labelled **(on laptop)** or **(on Kria)** to make the workflow unambiguous.

Target: Kria-PYNQ 3.0.1, Vitis AI 3.5, DPUCZDX8G ISA1 B4096 @ 300 MHz.

---

## 0. Prerequisites

**Hardware**

- Kria KV260 board + 12 V / 3 A power supply
- ≥32 GB microSD card + reader
- Ethernet cable (or USB-C UART for serial console)
- Host laptop running x86_64 Linux + Docker

**Software (laptop)**

- `dd` or balenaEtcher for SD flashing
- Docker (for AMD's Vitis AI compiler images)
- SSH client

---

## 1. Flash the SD card (on laptop)

Download the Kria-PYNQ image (v3.0.1 confirmed working with VAI 3.5):

```bash
# Image URL is on https://www.pynq.io/boards.html — look for KV260 v3.0.1
wget <kria-pynq-image-url> -O KV260_PYNQ.img.zip
unzip KV260_PYNQ.img.zip
sudo dd if=KV260_PYNQ.img of=/dev/sdX bs=4M status=progress conv=fsync
sync
```

Replace `/dev/sdX` with your SD card device (check with `lsblk` first —
**wrong target wipes your laptop drive**).

---

## 2. Boot and connect (on Kria → on laptop)

Insert SD, connect ethernet to your laptop or router, power on. First boot
takes ~5 minutes. Default credentials:

- Username: `ubuntu`
- Password: `xilinx`  (you'll be forced to change it on first login)

Find the Kria's IP:

```bash
# on laptop — assumes Kria is on the same subnet
nmap -sn 192.168.1.0/24 | grep -B2 -i kria
# or check the router DHCP table
```

SSH in:

```bash
# on laptop
ssh ubuntu@<kria-ip>
```

Set a hostname for convenience:

```bash
# on Kria
sudo hostnamectl set-hostname kria-lpr
```

---

## 3. Verify Vitis AI 3.5 stack and load DPU (on Kria)

The Kria-PYNQ 3.0.1 image ships with DPU-PYNQ. Confirm the libraries are at
3.5:

```bash
# on Kria
dpkg -l | grep -i 'vart\|xir\|vitis-ai-library\|target-factory' | awk '{print $2, $3}'
```

Expect `libvart`, `libxir`, `libvitis-ai-library`, `libtarget-factory` all
showing `3.5.0-1`. If any are missing or older, install:

```bash
# on Kria
sudo apt update
sudo apt install -y libvart-3.5 libxir-3.5 libvitis-ai-library-3.5 libtarget-factory-3.5
```

Load the DPU overlay and verify fingerprint:

```bash
# on Kria
sudo xmutil unloadapp
sudo xmutil loadapp kv260-benchmark
xdputil query | grep -i fingerprint
# Expected: 0x101000056010407
```

If the fingerprint matches, the DPU is ready.

---

## 4. Convert your model to xmodel (on laptop)

This is the model-development flow. Vitis AI's quantizer and compiler are
distributed as Docker images.

### 4a. Pull the right Vitis AI 3.5 image

Match your training framework:

```bash
# on laptop
docker pull xilinx/vitis-ai-pytorch-cpu:3.5.0       # for PyTorch
docker pull xilinx/vitis-ai-tensorflow2-cpu:3.5.0   # for TF2
docker pull xilinx/vitis-ai-tensorflow-cpu:3.5.0    # for TF1
```

### 4b. Run the container with your workspace mounted

```bash
# on laptop, from your model project directory
docker run -it --rm \
    -v "$(pwd)":/workspace \
    -w /workspace \
    xilinx/vitis-ai-pytorch-cpu:3.5.0 bash
```

Everything in your local directory now appears under `/workspace` inside
the container.

### 4c. Quantize (FP32 → INT8) — PyTorch example

Inside the container:

```bash
# in container
conda activate vitis-ai-pytorch

python -c "
import torch
from pytorch_nndct.apis import torch_quantizer
model = torch.load('my_model.pth').eval()
example_input = torch.randn(1, 3, 224, 224)
quantizer = torch_quantizer('calib', model, example_input, output_dir='quant_out')
# Run quantizer.quant_model(...) over your calibration dataset (~100-1000 images)
# then:
quantizer.export_quant_config()

quantizer = torch_quantizer('test', model, example_input, output_dir='quant_out')
quantizer.export_xmodel(output_dir='quant_out', deploy_check=False)
"
```

For TF2: `vai_q_tensorflow2 quantize --model my_model.h5 ...` — see AMD's
`vai_q_tensorflow2 --help` for the full flag set.

### 4d. Compile to xmodel

```bash
# in container
vai_c_xir \
    --xmodel quant_out/MyModel_int.xmodel \
    --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
    --output_dir compiled/ \
    --net_name my_model
```

The `arch.json` is for the KV260's `DPUCZDX8G_ISA1_B4096_0101000056010407`
fingerprint — same one the board reports in step 3.

Output: `compiled/my_model.xmodel`. Exit the container.

---

## 5. Deploy the xmodel (on laptop)

```bash
# on laptop
scp compiled/my_model.xmodel ubuntu@<kria-ip>:/home/ubuntu/
```

---

## 6. Run inference (on Kria)

Open Jupyter (`http://<kria-ip>:9090/lab`, password from step 2) or SSH +
Python. Minimal inference template:

```python
# on Kria
from pynq_dpu import DpuOverlay
import numpy as np
import cv2

overlay = DpuOverlay("dpu.bit")          # load the DPU overlay
overlay.load_model("/home/ubuntu/my_model.xmodel")
dpu = overlay.runner

# Allocate buffers from the model's declared tensor shapes
in_tensors  = dpu.get_input_tensors()
out_tensors = dpu.get_output_tensors()
in_shape  = tuple(in_tensors[0].dims)    # e.g. (1, 224, 224, 3) NHWC
input_data  = [np.empty(in_shape, dtype=np.float32, order='C')]
output_data = [np.empty(tuple(t.dims), dtype=np.float32, order='C') for t in out_tensors]

# Preprocess an image to the model's expected input format
img = cv2.imread('test.jpg')                  # BGR uint8 HWC
img = cv2.resize(img, (in_shape[2], in_shape[1]))
img = img.astype(np.float32) - np.array([104, 117, 123])  # mean-subtract (varies per model)
input_data[0][0] = img

# Run
dpu.wait(dpu.execute_async(input_data, output_data))

# Inspect output
print("output shape:", output_data[0].shape)
print("top-1 class:", int(output_data[0].flatten().argmax()))
```

**Critical**: the `mean` and `scale` values in preprocessing must match
what AMD used during quantization calibration — read them from the
`.prototxt` file that ships alongside the xmodel, not from training-time
defaults. Wrong preprocessing values cost 10–20% top-1 accuracy with no
visible error.

---

## 7. Next steps

- For an end-to-end benchmark suite over multiple models (latency, power,
  accuracy, mAP), see `notebooks/04_vai35_benchmark.ipynb`.
- For live camera inference, see `notebooks/07_kv260_live_inference_v2.ipynb`.
- For the model zoo's pre-compiled xmodels (skip step 4 entirely), see
  AMD's Vitis AI Model Zoo at github.com/Xilinx/Vitis-AI/tree/master/model_zoo.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `xdputil query` says "no device" | DPU overlay not loaded | `sudo xmutil loadapp kv260-benchmark` |
| Fingerprint mismatch | Wrong overlay (.bit) loaded | Re-run step 3 with correct app name |
| `RuntimeError: ... fingerprint mismatch` from PYNQ | xmodel compiled for wrong DPU | Recompile with matching `arch.json` |
| Inference runs but accuracy ~5% | Preprocessing mean/scale wrong | Read values from model's `.prototxt`, not defaults |
| `cv2.VideoCapture` returns False | Camera permissions or wrong device | `sudo chmod 666 /dev/video0`, check `v4l2-ctl --list-devices` |
