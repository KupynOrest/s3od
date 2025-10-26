<div align="center">

# S3OD: Towards Generalizable Salient Object Detection with Synthetic Data

[**Orest Kupyn**](https://github.com/KupynOrest)<sup>1</sup> · [**Hirokatsu Kataoka**](https://hirokatsukataoka.net/)<sup>12</sup> · [**Christian Rupprecht**](https://chrirupp.github.io/)<sup>1</sup> ·

<sup>1</sup>University of Oxford · <sup>2</sup>AIST, Japan

<a href='#'><img src='https://img.shields.io/badge/arXiv-Coming Soon-red'></a>
<a href='https://huggingface.co/okupyn/s3od'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Model-blue'></a>
<a href='https://huggingface.co/datasets/okupyn/s3od_dataset'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-blue'></a>
[![PyPI](https://img.shields.io/badge/PyPI-s3od-blue)](https://pypi.org/project/s3od/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**S3OD** is a large-scale fully synthetic dataset for salient object detection and background removal, with 140K high-quality images generated using diffusion models. Our model, trained on large-scale synthetic data and fine-tuned on real, achieves state-of-the-art performance on real-world benchmarks and enables single-step effective background removal from images in high resolution.

![banner](./images/banner.png)

## News
- [2025/10/26] 🔥 Release Version 0.1.0 - Training code, synthetic dataset, and inference package!

## S3OD Dataset Download Instructions

The S3OD dataset contains 140K+ synthetic images with high-quality masks for salient object detection.

### Using HuggingFace Datasets Library (Recommended)

The simplest way to use the dataset is through the `datasets` library:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("okupyn/s3od_dataset", split="train")

# Access data
for sample in dataset:
    image = sample["image"]      # PIL Image
    mask = sample["mask"]        # PIL Image (mask)
    caption = sample["caption"]  # str
    category = sample["category"]# str
    image_id = sample["id"]      # str

# Or access specific samples
sample = dataset[0]
image = sample["image"]
mask = sample["mask"]
```

### Dataset Statistics

- **Total images**: 140,000+
- **Categories**: 1,000+ ImageNet classes
- **Resolution**: Variable (resized during training)
- **Format**: JPEG (images), PNG (masks)
- **Storage size**: ~35GB (Parquet format)

### Dataset Structure

Each sample in the dataset contains:
- `image`: RGB image (PIL Image)
- `mask`: Binary segmentation mask (PIL Image)
- `caption`: Descriptive caption generated for the image
- `category`: Object category from ImageNet
- `id`: Unique image identifier

## Installation

#### Create a Conda virtual environment

```bash
conda create --name s3od python=3.10
conda activate s3od
```

#### Install the inference package

```bash
pip install s3od
```

#### Or clone for training/research

```bash
git clone https://github.com/organization/s3od.git
cd s3od
pip install -e ./
```

## Usage

### Quick Start

```python
from s3od import BackgroundRemoval
from PIL import Image

# Initialize detector (automatically downloads model from HuggingFace)
detector = BackgroundRemoval()

# Load and process image
image = Image.open("your_image.jpg")
result = detector.remove_background(image)

# Save result with transparent background
result.rgba_image.save("output.png")

# Access predictions
best_mask = result.predicted_mask  # Best mask (H, W) numpy array
all_masks = result.all_masks       # All masks (N, H, W) numpy array  
all_ious = result.all_ious         # IoU scores (N,) numpy array
```

## Demo

**🌐 Online Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/okupyn/s3od)

**💻 Run Locally:**

```bash
cd demo
pip install -r requirements.txt
python app.py
```

## Training

S3OD uses [Hydra](https://hydra.cc/) for configuration management and [PyTorch Lightning](https://lightning.ai/) for training.

### Training on Synthetic Data

```bash
# Single GPU
python -m synth_sod.model_training.train \
    dataset=synth \
    model=dinob \
    backend=1gpu

# Multi-GPU (DDP)
torchrun --standalone --nnodes=1 --nproc_per_node=NUM_GPUS \
    -m synth_sod.model_training.train \
    dataset=synth \
    model=dinob \
    backend=NUM_GPUs
```

Replace `NUM_GPUS` with the number of GPUs you want to use (e.g., 2, 4, 8).

### Available Configurations

Check `synth_sod/model_training/config/` for available options:

**Models** (`model=...`):
- `dinob`: DINOv3-Base (default)
- `dinol`: DINOv3-Large
- `flux_teacher`: FLUX-enhanced teacher model

**Datasets** (`dataset=...`):
- `synth`: Synthetic S3OD dataset
- `duts`: DUTS dataset
- `dis`: DIS5K dataset
- `full`: All datasets combined

**Compute** (`backend=...`):
- `1gpu`, `2gpu`, `4gpu`, `8gpu`

### Training Examples

```bash
# Train on real-world benchmark
python -m synth_sod.model_training.train dataset=duts model=dinob

# Train teacher model with FLUX features
python -m synth_sod.model_training.train \
    -cn train_teacher \
    dataset=full \
    model=flux_teacher \
    backend=4gpu

# Resume training from checkpoint
python -m synth_sod.model_training.train \
    dataset=synth \
    model=dinob \
    training_hyperparams.resume=True
```

## Generating Synthetic Data

Our data generation pipeline uses FLUX diffusion models and concept-guided attention to create high-quality synthetic training data.

### Prerequisites

1. FLUX model weights
2. OpenAI API key (for caption generation)

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Generation Pipeline

1. **Generate Captions & Tags**
```bash
cd synth_sod/data_generation/flux_finetune
python generate_captions.py --image_dir <dir> --output_file captions.json
python tag_data.py --image_dir <dir> --output_file tags.json
```

2. **Extract FLUX Features**
```bash
python feature_extraction.py \
    --caption_file captions.json \
    --tag_file tags.json \
    --save_folder <features_dir> \
    --model_path <flux_model_path>
```

3. **Generate Images and Masks**
```bash
python generate_train_images.py --config_path generation_config.yaml
```

4. **Filter Dataset** (optional)
```bash
python run_filtering.py --config_path filtering_config.yaml
```

Update configuration files with your paths before running. See example configs in `synth_sod/data_generation/`.

## Cite

If you use S3OD in your research, please cite:

```bibtex
@article{s3od2025,
  title={S3OD: Synthetic Salient Object Detection},
  author={Kataoka, Hirokatsu and Kupyn, Orest and others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
