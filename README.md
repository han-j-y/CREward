<div align="center">

# CREward: A Type-Specific Creativity Reward Model

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2511.19995)
[![Project Page](https://img.shields.io/badge/Project-Page-4285F4?logo=googlechrome&logoColor=white)](https://han-j-y.github.io/creward_prj/)
[![Dataset](https://img.shields.io/badge/Dataset-CREBench-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/hanjy/CREBench)

<br>

**Official PyTorch implementation**

<br>

**Jiyeon Han** · **Ali Mahdavi-Amiri** · **Hao Zhang** · **Haedong Jeong**

<br>

</div>

---

## Abstract 

Creativity is a complex phenomenon. When it comes to representing and assessing creativity, treating it as a single undifferentiated quantity would appear naive and underwhelming. In this work, we learn the \emph{first type-specific creativity reward model}, coined CREward, which spans three creativity ``axes," geometry, material, and texture, to allow us to view creativity through the lens of the image formation pipeline. To build our reward model, we first conduct a human benchmark evaluation to capture human perception of creativity for each type across various creative images. We then analyze the correlation between human judgments and predictions by large vision-language models (LVLMs), confirming that LVLMs exhibit strong alignment with human perception. Building on this observation, we collect LVLM-generated labels to train our CREward model that is applicable to both evaluation and generation of creative images. We explore three applications of CREward: creativity assessment, explainable creativity, and creative sample acquisition for both human design inspiration and guiding creative generation through low-rank adaptation.

<div align="center">
  <img src="./main.png" alt="CREward teaser" width="100%">
</div>

---

## Overview

CREward is a reward model that scores the creativity of generated images along four type-specific dimensions: **geometry**, **material**, **texture**, and **overall** appearance. A frozen vision backbone extracts image features; a lightweight MLP head (`reward_head`) maps features to four continuous scores. The default configuration uses the SigLIP vision tower of [Gemma-3-4B-IT](https://huggingface.co/google/gemma-3-4b-it).

This repository provides:

- Model definition and inference (`src/models.py`)
- Pre-trained reward-head weights (`ckpt/`)
- Pairwise dataset utilities for CREBench (`src/utils.py`)
- An example notebook ([`CREward_eg.ipynb`](CREward_eg.ipynb))

---

## Usage

### Setup

**Requirements:** Python 3.10+, NVIDIA GPU with CUDA (recommended for the default backbone).

**Tested stack:** Python 3.13.9 · PyTorch 2.8.0 (CUDA 12.9) · see [`requirements.txt`](requirements.txt)

Clone the repository:

```bash
git clone https://github.com/han-j-y/CREward.git
cd CREward
```

#### Optional: Conda (recommended)

```bash
conda create -n creward python=3.13 -y
conda activate creward
```



#### Install pip only

```bash
pip install -r requirements.txt
```


Run notebooks and scripts from the **repository root** (the folder containing `src/` and `ckpt/`).

### Inference

```python
import torch
from PIL import Image
from src.models import CreativityReward

device = "cuda"
backbone = "siglip-gemma3"

model = CreativityReward(device=device, backbone=backbone)
model.reward_head.load_state_dict(
    torch.load("ckpt/siglip-gemma3/reward_head_best.ckpt", map_location=device),
    strict=True,
)
model.eval()

image = Image.open("test_images/chair_0.png")
with torch.no_grad():
    reward = model.compute_reward(image)  # [1, 4]

print(reward)
```


### Visualization

Run [`CREward_eg.ipynb`](CREward_eg.ipynb) for an end-to-end example: load the model, score two test images, and plot per-dimension horizontal bar charts.

---


### Pre-trained Models

Vision backbones are loaded from Hugging Face at runtime. The trainable **reward head** must be placed under `ckpt/<backbone>/reward_head_best.ckpt` (see table below).

Reward-head checkpoints are hosted on [jhan/CREward](https://huggingface.co/jhan/CREward/tree/main). Download with `huggingface-cli` or your browser, then copy into `ckpt/`:

```bash
huggingface-cli download jhan/CREward siglip-gemma3/reward_head_best.ckpt --local-dir ckpt
# repeat for other backbones, or download the full repo
```

**Reward dimensions** (output index order of `compute_reward`):

| Index | Dimension |
|-------|-----------|
| 0 | Geometry |
| 1 | Material |
| 2 | Texture |
| 3 | Overall |

**Vision backbones** supported in code (`CreativityReward(..., backbone=...)`):

| `backbone` | Encoder | Local reward head | Download |
|------------|---------|-------------------|----------|
| `siglip-gemma3` | [Gemma-3-4B-IT](https://huggingface.co/google/gemma-3-4b-it) vision tower (default) | `ckpt/siglip-gemma3/reward_head_best.ckpt` | [reward head](https://huggingface.co/jhan/CREward/blob/main/siglip-gemma3/reward_head_best.ckpt) |
| `siglip2` | [SigLIP 2 SO400M](https://huggingface.co/google/siglip2-so400m-patch14-384) | `ckpt/siglip2/reward_head_best.ckpt` | [reward head](https://huggingface.co/jhan/CREward/blob/main/siglip2/reward_head_best.ckpt) |
| `vgg16` | VGG16 (ImageNet via `torchvision`) | `ckpt/vgg16/reward_head_best.ckpt` | [reward head](https://huggingface.co/jhan/CREward/blob/main/vgg16/reward_head_best.ckpt) |
| `clip` | [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14) | `ckpt/clip/reward_head_best.ckpt` | [reward head](https://huggingface.co/jhan/CREward/blob/main/clip/reward_head_best.ckpt) |
| `dreamsim` | [DreamSim](https://github.com/ssundaram21/dreamsim) | `ckpt/dreamsim/reward_head_best.ckpt` | [reward head](https://huggingface.co/jhan/CREward/blob/main/dreamsim/reward_head_best.ckpt) |
| `dino3` | [DINOv3 ViT-H/16+](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) | `ckpt/dino3/reward_head_best.ckpt` | [reward head](https://huggingface.co/jhan/CREward/blob/main/dino3/reward_head_best.ckpt) |

Each backbone needs the matching reward head; vision weights for non-VGG encoders are fetched automatically by `transformers` / `dreamsim` on first use.

---



## Repository Structure

```
.
├── README.md
├── requirements.txt
├── CREward_eg.ipynb
├── ckpt/
│   └── siglip-gemma3/
│       └── reward_head_best.ckpt
├── src/
│   ├── __init__.py
│   ├── models.py          # CreativityReward
│   └── utils.py           # CreRewardDataset, dataset_preprocess
└── test_images/
```

**Import:**

```python
from src.models import CreativityReward
```

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{han2026creward,
  title={CREward: A Type-Specific Creativity Reward Model},
  author={Han, Jiyeon and Mahdavi-Amiri, Ali and Zhang, Hao and Jeong, Haedong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21932--21941},
  year={2026}
}
```

---

## Acknowledgements

We thank the authors of [Gemma](https://huggingface.co/google/gemma-3-4b-it) and [ImageReward](https://github.com/zai-org/ImageReward) the open-source vision backbones used in our ablations. This implementation builds on [PyTorch](https://pytorch.org/) and [Hugging Face Transformers](https://github.com/huggingface/transformers).

