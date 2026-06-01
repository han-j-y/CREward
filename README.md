# CREward: A Type-Specific Creativity Reward Model

**Official PyTorch implementation of the CVPR 2026 paper *CREward: A Type-Specific Creativity Reward Model*.**

[Paper](https://arxiv.org/pdf/2511.19995) | [Project Page](https://han-j-y.github.io/creward_prj/) | [arXiv](https://arxiv.org/abs/2511.19995) | [Dataset](https://huggingface.co/datasets/hanjy/CREBench)

**Jiyeon Han**, **Ali Mahdavi-Amiri**, **Hao Zhang**, **Haedong Jeong**

---

## Abstract 

Creativity is a complex phenomenon. When it comes to representing and assessing creativity, treating it as a single undifferentiated quantity would appear naive and underwhelming. In this work, we learn the \emph{first type-specific creativity reward model}, coined CREward, which spans three creativity ``axes," geometry, material, and texture, to allow us to view creativity through the lens of the image formation pipeline. To build our reward model, we first conduct a human benchmark evaluation to capture human perception of creativity for each type across various creative images. We then analyze the correlation between human judgments and predictions by large vision-language models (LVLMs), confirming that LVLMs exhibit strong alignment with human perception. Building on this observation, we collect LVLM-generated labels to train our CREward model that is applicable to both evaluation and generation of creative images. We explore three applications of CREward: creativity assessment, explainable creativity, and creative sample acquisition for both human design inspiration and guiding creative generation through low-rank adaptation.

![image](./main.png)

---

## Overview

CREward is a reward model that scores the creativity of generated images along four type-specific dimensions: **geometry**, **material**, **texture**, and **overall** appearance. A frozen vision backbone extracts image features; a lightweight MLP head (`reward_head`) maps features to four continuous scores. The default configuration uses the SigLIP vision tower of [Gemma-3-4B-IT](https://huggingface.co/google/gemma-3-4b-it).

This repository provides:

- Model definition and inference (`src/models.py`)
- Pre-trained reward-head weights (`ckpt/`)
- Pairwise dataset utilities for CREBench (`src/utils.py`)
- An example notebook ([`CREward_eg.ipynb`](CREward_eg.ipynb))

---

## Table of Contents

- [Environment](#environment)
- [Pre-trained Models](#pre-trained-models)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Pairwise forward](#pairwise-forward)
  - [Visualization](#visualization)
- [Training Utilities](#training-utilities)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Environment

**Requirements:** Python 3.10+, CUDA GPU (recommended for `siglip-gemma3`).

### Tested configuration

We verified inference with the following stack (see [`requirements.txt`](requirements.txt)):

| Package | Version |
|---------|---------|
| Python | 3.13.9 |
| PyTorch | 2.8.0 (CUDA 12.9) |
| torchvision | 0.23.0 |
| transformers | 4.56.0 |
| accelerate | 1.12.0 |
| huggingface-hub | 0.36.0 |
| pillow | 12.0.0 |
| numpy | 2.3.5 |
| matplotlib | 3.10.7 |

Minor version differences are usually fine; pin exact versions if you need strict reproducibility.

### Setup

```bash
git clone https://github.com/<ORG>/CREward.git
cd CREward

# PyTorch with CUDA: use https://pytorch.org/get-started/locally/ if needed, then:
pip install -r requirements.txt
```

Optional dependency for `backbone='dreamsim'`:

```bash
pip install -r requirements-optional.txt
```

The default backbone downloads `google/gemma-3-4b-it` from Hugging Face. For gated models:

```bash
huggingface-cli login
```

All commands below assume the **repository root** as the working directory.

---

## Pre-trained Models

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

## Data Preparation

CREBench is the pairwise creativity preference benchmark used for training and evaluation in our paper. It is hosted on Hugging Face:

**[hanjy/CREBench](https://huggingface.co/datasets/hanjy/CREBench)**

| Split | Subset | # Pairs | Labels |
|-------|--------|---------|--------|
| LVLM | `CREBench_LVLM` | 5,000 (1,000 × 5 categories) | Gemma-3-27B-IT, shape `(N, 4)` |
| Human | `CREBench_Human` | 500 (100 × 5 categories) | 5 raters, shape `(5, N, 4)` |

**Categories:** `chair`, `bowl`, `car`, `vase`, `handbag`.

Download:

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="hanjy/CREBench",
    repo_type="dataset",
    local_dir="./CREBench",
)
```

Images are named `pair_{idx:04d}_0.png` and `pair_{idx:04d}_1.png` per category folder. Labels are stored as `.pkl` files under `Label/`; see the [dataset card](https://huggingface.co/datasets/hanjy/CREBench) for the full schema.

---

## Usage

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

### Pairwise forward

For a batch of image pairs (training or ranking):

```python
# x1, x2: [B, C, H, W], same device as model
out = model(x1, x2)  # [B, 2, 4]
```

### Visualization

Run [`CREward_eg.ipynb`](CREward_eg.ipynb) for an end-to-end example: load the model, score two test images, and plot per-dimension horizontal bar charts.

---

## Training Utilities

Pairwise data loading and split construction:

```python
from src.utils import CreRewardDataset, dataset_preprocess, param_size_BM, make_dir
```

`dataset_preprocess` expects category folders of paired PNGs and NumPy ranking files; see `src/utils.py` for arguments and directory layout. Training scripts will be released in a future update.

---

## Repository Structure

```
.
├── README.md
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

Please also cite CREBench when using the benchmark data.

---

## Acknowledgements

We thank the authors of [Gemma](https://huggingface.co/google/gemma-3-4b-it) and the open-source vision backbones used in our ablations. This implementation builds on [PyTorch](https://pytorch.org/) and [Hugging Face Transformers](https://github.com/huggingface/transformers).

---

## License

This repository is released under the [LICENSE](LICENSE) file in the project root. Third-party weights (e.g., Gemma-3-4B-IT) are subject to their respective licenses on Hugging Face.

For questions or issues, please open a GitHub issue.
