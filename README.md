# SciPostLayoutTree: Supplementary Code

This repository contains the supplementary code for our paper:

> **"SciPostLayoutTree: A Dataset for Structural Analysis of Scientific Posters"**

---

## 📦 Included Contents

- Source code for training, evaluation, visualization, and analysis
- Shell scripts for automated experiment execution
- Jupyter notebooks for data and model analysis

---

## 📁 Dataset Construction

### SciPostLayout (base dataset)

Our annotation builds upon the publicly released **SciPostLayout** dataset:

- Tanaka et al., *SciPostLayout: A Dataset for Layout Analysis and Layout Generation of Scientific Posters*  
  [arXiv:2407.19787](https://arxiv.org/abs/2407.19787)

To prepare the data:

1. Download SciPostLayout from Hugging Face:  
   [https://huggingface.co/datasets/omron-sinicx/scipostlayout_v2](https://huggingface.co/datasets/omron-sinicx/scipostlayout_v2)

2. Extract it so that the structure becomes:

   ```
   ./../scipostlayout/poster/
   ```

### SciPostLayoutTree (our dataset)

3. Download our annotation files (`*_tree_ocr.json`) from Hugging Face:
    [https://huggingface.co/datasets/omron-sinicx/scipostlayouttree](https://huggingface.co/datasets/omron-sinicx/scipostlayouttree)

4. After downloading, place the annotation JSONs as follows:

    ```bash
    mv train_tree_ocr.json ./../scipostlayout/poster/
    mv dev_tree_ocr.json   ./../scipostlayout/poster/
    mv test_tree_ocr.json  ./../scipostlayout/poster/
    ```

---

### DocHieNet (comparison dataset)

We also use **DocHieNet** for a comparative analysis:

- Xing et al., *DocHieNet: A Large and Diverse Dataset for Document Hierarchy Parsing*  
  [EMNLP 2024, ACL Anthology](https://aclanthology.org/2024.emnlp-main.65/)

#### 📥 Download and Extraction Instructions

```bash
mkdir ./../scipostlayout/DocHieNet/
cd ./../scipostlayout/DocHieNet/
```

1. Download from ModelScope:  
   [https://modelscope.cn/datasets/iic/DocHieNet](https://modelscope.cn/datasets/iic/DocHieNet)

2. If split into parts:

```bash
cat dochienet_dataset.zip.part-* > dochienet_dataset.zip
unzip dochienet_dataset.zip
```

> 🔹 DocHieNet is not included in this archive.

---

## 🖼️ Visualizing Annotations

You can visualize tree annotations using:

```bash
pip install opencv-python numpy matplotlib tqdm
python visualize_annotation.py
```

---

## 💻 Experimental Environment

- **GPUs**: 8 × NVIDIA A100-SXM4-80GB
- **CPUs**: 2 × AMD EPYC 7713 (128 cores)
- **RAM**: 2.0 TiB (1.9 TiB usable)
- **OS**: Ubuntu 22.04.5 LTS
- **Driver / CUDA**: 535.161.08 / CUDA 12.2
- **Python**: 3.10.10
- **Key Libraries**:
  - PyTorch 2.5.1+cu121
  - Transformers 4.52.3
  - Detectron2 0.6
  - Detrex 0.3.0

---

## 🧩 Pretrained Checkpoints

Please manually download and place the following pretrained models into the `./detrex` directory:

| Backbone     | Filename | Download Link |
|--------------|----------|----------------|
| ResNet-50    | `dino_r50_4scale_24ep.pth` | [Download](https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_r50_4scale_24ep.pth) |
| ViT          | `dino_vitdet_base_4scale_50ep.pth` | [Download](https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_vitdet_base_4scale_50ep.pth) |
| Swin         | `dino_swin_base_384_4scale_12ep.pth` | [Download](https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_base_384_4scale_12ep.pth) |
| InternImage  | `dino_internimage_base_4scale_12ep.pth` | [Download](https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/dino_internimage_base_4scale_12ep.pth) |
| DiT          | `dit-base-224-p16-500k.pth` (rename if needed) | [Download](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ERfMnkSl2mZFlv68E6xxhH4BvJVQOYcwpna3QMbbbLDQCA?e=tOPZRx) |

---

## 🐳 Docker Setup

```bash
bash run_docker.sh

apt update -y
apt upgrade -y
apt install -y \
  make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl \
  llvm libncursesw5-dev xz-utils tk-dev \
  libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
  git ca-certificates ninja-build

curl https://pyenv.run | bash

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv --version
```

You can save a running container as a Docker image and reuse it later.

```bash
docker commit scipostlayouttree scipostlayouttree
docker run --shm-size=64g -p 8888:8888 -it --rm --gpus all --name scipostlayouttree \
 --volume $(pwd):/scipostlayouttree_code --volume $(pwd)/../scipostlayout:/scipostlayout scipostlayouttree
```

---

## 🧱 Python Environment Setup

```bash
cd /scipostlayouttree_code/detrex
pyenv install 3.10.10
pyenv global 3.10.10
python -m venv venv-detrex
source venv-detrex/bin/activate
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install -e .
wandb init
```

---

## 🌀 Shell Scripts

- `run.sh`:  
  Train a model using a specified method and backbone  
  Example: `METHOD="DRGGBBoxEmbTFEnc"` = DRGG-BE  
  `METHOD="DRGGBBoxEmbTextEmbTFEnc"` = DRGG-BETE  
  Use `BEAM_WIDTH=20` for DRGG-BEBS or DRGG-BETEBS

- `run_loop.sh`:  
  Loop over all backbone x method combinations

- `run_loop_inference.sh`:  
  Run inference with beam width = 1 and 20

- `run_notebook.sh`:  
  Launch Jupyter Notebook server

---

## 📘 Code Overview

| File | Description |
|------|-------------|
| `analyze_dataset.ipynb` | Analysis of the SciPostLayoutTree dataset |
| `analyze_dochienet.ipynb` | Comparison with DocHieNet |
| `create_tables.ipynb` | Generates paper figures and tables from inference results |
| `visualize_prediction.py` | Visualizes prediction results |
| `projects/relation/modeling/drgg.py` | DRGG and variant models |
| `projects/relation/modeling/gtbbox_relation_predictor.py` | Full Layout Tree Decoder implementation |
| `projects/relation/configs/*.py` | Configuration files per backbone |
| `configs/common/data/tree_evaluator.py` | Compute STEDS, REDS, and TED |

---
