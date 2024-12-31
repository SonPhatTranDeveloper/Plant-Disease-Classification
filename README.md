# Low-Resource Plant Disease Classification with Foundation Models

## Overview
This repository contains the official implementation of our research paper "Low-Resource Plant Disease Classification with Foundation Models". We explore how various foundation models perform on plant disease classification tasks in low-resource settings, comparing traditional CNN approaches with modern foundation models like CLIP, BLIP, and DINOv2.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Models](#models)
  - [CNN Baseline](#cnn-baseline)
  - [CLIP](#clip)
  - [BLIP](#blip)
  - [DINOv2](#dinov2)
- [Results](#results)
- [License](#license)

## Dataset

### PlantVillage Dataset Overview
The PlantVillage dataset is a comprehensive collection of plant disease images that has become a standard benchmark in agricultural computer vision. It contains 54,306 images of healthy and diseased plant leaves spanning 38 different classes (14 crop species and 26 diseases). The images are taken in controlled laboratory conditions against uniform backgrounds, making it an excellent starting point for plant disease classification research.

Key features of the dataset:
- 38 classes covering different crop-disease combinations
- High-quality RGB images (256 x 256 pixels)
- Controlled imaging conditions
- Expert-validated labels
- Multiple images per plant showing different disease stages and variations

### Low-Resource Adaptation
While the full PlantVillage dataset contains thousands of images, real-world scenarios often face severe data limitations, especially for rare plant diseases or newly emerging pathogens. To simulate these low-resource conditions, we create a challenging subset of the data:

1. Training set: 10 images per class randomly sampled from the original training split
2. Validation set: 20 images per class
3. Test set: All remaining images

This setup creates a realistic low-resource scenario where:
- Models must learn from very limited examples (10 shots)
- The validation set is kept small to reflect real-world constraints
- A large test set ensures reliable evaluation of generalization

## Installation

### Prerequisites
- Python 3.11
- CUDA-compatible GPU (recommended)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/low-resource-plant-disease.git
cd low-resource-plant-disease
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set Python path:
```bash
export PYTHONPATH=.
```

## Dataset Preparation

### Download
1. Download the PlantVillage dataset from [Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1) (non-augmented version)
2. Extract to `raw_dataset` folder with the following structure:
```
raw_dataset/
└── full/
    ├── disease_1/
    │   ├── Image1.png
    │   ├── Image2.png
    │   └── ...
    ├── disease_2/
    │   ├── Image3.png
    │   ├── Image4.png
    │   └── ...
    └── ...
```

### Dataset Processing
1. Rename disease folders according to the standardized format (see [disease_naming.txt](docs/disease_naming.txt))
2. Create low-resource dataset:
```bash
python scripts/extract_small_dataset.py
```

## Models

### CNN Baseline
Train the baseline CNN model:
```bash
python models/baseline_cnn/train.py
```

### CLIP
Three training approaches available:

1. One-shot classification:
```bash
python models/clip/one_shot.py
```

2. End-to-end fine-tuning (text & image matching):
```bash
python models/clip/fine_tuning_end_to_end.py
```

3. Classification head fine-tuning:
```bash
python models/clip/fine_tuning_classification.py
```

### BLIP
Three training approaches available:

1. One-shot classification:
```bash
python models/blip/one_shot.py
```

2. End-to-end fine-tuning:
```bash
python models/blip/fine_tuning_end_to_end.py
```

3. Classification head fine-tuning (upcoming):
```bash
python models/blip/fine_tuning_classification.py  # To be implemented
```

### DINOv2

#### Setup
1. Download `ViT-L/14 distilled` weights from the [official DINOv2 repository](https://github.com/facebookresearch/dinov2)
2. Place weights in `models/dinov2/weights/`

#### Training Options
1. One-shot classification (k-NN):
```bash
python models/dinov2/one_shot.py
```

2. Fine-tuning with basic augmentation:
```bash
python models/dinov2/fine_tuning_classification.py
```

3. Fine-tuning with CutMix:
```bash
python models/dinov2/fine_tuning_cut_mix.py
```

4. Fine-tuning with MixUp:
```bash
python models/dinov2/fine_tuning_mix_up.py
```

## Results
(Will be added)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- PlantVillage dataset creators
- Foundation model developers (CLIP, BLIP, DINOv2)