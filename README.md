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

Vision backbones are loaded from Hugging Face at runtime. The trainable **reward head** must be placed locally as follows:

| Component | Path | Description |
|-----------|------|-------------|
| Reward head (Gemma-SigLIP) | `ckpt/siglip-gemma3/reward_head_best.ckpt` | CREward head trained with `siglip-gemma3` backbone |
| Vision backbone | — | `google/gemma-3-4b-it` (auto-downloaded via `transformers`) |

Download `reward_head_best.ckpt` from the [GitHub Releases](https://github.com/<ORG>/CREward/releases) page if it is not included in your clone.

**Reward dimensions** (output index order of `compute_reward`):

| Index | Dimension |
|-------|-----------|
| 0 | Geometry |
| 1 | Material |
| 2 | Texture |
| 3 | Overall |

**Optional vision backbones** supported in code (`CreativityReward(..., backbone=...)`):

| `backbone` | Encoder |
|------------|---------|
| `siglip-gemma3` | Gemma-3-4B-IT vision tower (default) |
| `siglip2` | SigLIP 2 SO400M |
| `vgg16` | VGG16 |
| `clip` | CLIP ViT-L/14 |
| `dreamsim` | DreamSim |
| `dino3` | DINOv3 ViT-H/16+ |

Each backbone requires a reward head trained for that encoder; only the Gemma-SigLIP checkpoint is provided above.

---



## Repository Structure

```
.
├── README.md
├── environment.yml
├── requirements.txt
├── requirements-optional.txt
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

