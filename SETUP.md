
# Cosmos Supervisor — Setup Instructions

This project uses NVIDIA Cosmos Reason2 for hierarchical vision-based robot supervision.

The repository does not vendor the model code directly.
Instead, it runs on top of the official Cosmos Reason2 environment.

---

## 1️⃣ System Requirements

### Recommended environment

Ubuntu 22.04 or similar Linux distribution

NVIDIA GPU with CUDA 12.x driver support

Python 3.10+ (tested with Python 3.12)

24 GB+ GPU memory recommended

Works on most CUDA-enabled Linux systems.

### Tested environments

GCP Deep Learning VM

CUDA-enabled Ubuntu workstation (Ubuntu 24.04)

NVIDIA GPU with CUDA 12.8 driver

PyTorch with CUDA support

GPU required. CPU-only inference (Mac) is too slow for practical use.

---

## 2️⃣ Sanity Check GPU

```bash
nvidia-smi
```

Ensure CUDA is available and the GPU is visible.

---

## 3️⃣ Install uv (One-Time)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc || true
source $HOME/.local/bin/env || true
```

Verify:

```bash
uv --version
```

---

## 4️⃣ Clone Cosmos Reason2

```bash
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git
cd cosmos-reason2
```

---

## 5️⃣ Create Virtual Environment

```bash
uv venv
source .venv/bin/activate
```

---

## 6️⃣ Install Dependencies

Cosmos Reason2 uses pyproject.toml + uv.lock.

```bash
uv sync
```

---

## 7️⃣ Login to Hugging Face (Required for Model Weights)

```bash
huggingface-cli login
```

Enter Token when prompted

You must have access to gated Cosmos Reason2 weights.

Get access from huggingface website.
   
https://huggingface.co/nvidia/Cosmos-Reason2-2B
   
https://huggingface.co/nvidia/Cosmos-Reason2-8B

---

## 8️⃣ Install Torch + Torchvision (CUDA 12.1)

```bash
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 9️⃣ Verify CUDA + Torch

```bash
python - << 'EOF'
import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
EOF
```

`cuda available` must be **True**.

---



## 🔟 Clone This Project

```
cd ~
git clone https://github.com/AIBotTeachesAI/cosmos-robot-supervisor.git

```

This repository contains:

hierarchical supervision pipeline

sample videos

reproducible outputs


## 1️⃣1️⃣ Activate Cosmos Environment (if not already done)

```bash
cd ~/cosmos-reason2
source .venv/bin/activate
```


# 🚀 Approach 1 — Full Video Reasoning

### Run Level‑1

```bash
cd ~/cosmos-robot-supervisor
python src/approach1/run_level1_all.py
```

(It will download the cosmos reason model files the first time)


#### Deprecation warnings during Level 1 execution


**Note:** You may see the following warnings during execution - these are harmless and can be safely ignored:
- `torchcodec` is not installed, falling back to `torchvision`
- `torchvision` decoding is deprecated
- Generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']

As long as you see output like `SUCCESS_LIFT (conf=0.99)`, the pipeline is working correctly.

### Run Level‑2

```bash
cd ~/cosmos-robot-supervisor
python src/approach1/level2_from_fullvideo_files.py
```

---

# 🔬 Approach 2 — Clip‑Level Reasoning

## Install ffmpeg (one time)


 ```bash
 sudo apt-get update
``` 

Note: This could take a few minutes


```bash
sudo apt-get install -y ffmpeg
```

Note: This could take a few minutes


### Level‑0. Generate sliding clips

```bash
cd ~/cosmos-robot-supervisor
bash src/approach2_exploration/video_utils/make_slowmo_and_sliding_clips.sh
```

### Level‑1. Describe clip states

```bash
cd ~/cosmos-robot-supervisor
python src/approach2_exploration/level1_explain_batch_state_label.py --fps 8
```

**Note:** You will see the same deprecation warnings as in Approach 1 (`torchcodec`, `torchvision`, generation flags). These are harmless and can be safely ignored. The script will process clips for each scenario and save results to `outputs/approach2_exploration/`.


### Level‑2. reasoning

```bash
cd ~/cosmos-robot-supervisor
python src/approach2_exploration/level2_from_label_files.py
```

---

# Notes

- This project focuses on reasoning supervision, not motor control.
- No policy training or fine‑tuning is performed.
- Cosmos Reason2 is used for structured vision‑based reasoning.
- Approach 2 explores clip‑level temporal reasoning and aggregation.

---

# Troubleshooting

### CUDA not detected
Ensure:
- NVIDIA driver is installed
- CUDA version matches torch wheel
- The virtual environment is activated

### Hugging Face access denied
You must:
- Request access to Cosmos Reason2
- Accept the model license on Hugging Face
- Re‑login using `huggingface-cli login`

---

