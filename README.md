# Low-resource Plant disease Classification

## About
This is the code for my upcoming research paper "Low-resource Plant Diseoutside of the ase classification with Foundation Models".

## How to run

Firstly, run the following code before you start any training
```bash
export PYTHONPATH=.
```

### Preparing the dataset

Firstly, download the dataset from https://data.mendeley.com/datasets/tywbtsjrjv/1 (without augmentation) and extract it
into ```raw_dataset``` folder with the following format:
```
raw_dataset
    full
        disease_1
            Image1.png
            Image2.png
            ...
        disease_2
            Image3.png
            Image4.png
            ...
```

Rename the diseases (folders) using the following format

```
Apple__apple_scab
Apple__black_rot
Apple__cedar_apple_rust
Apple__healthy
Background
Blueberry__healthy
Cherry__healthy
Cherry__powdery_mildew
Corn__cercospora_leaf_spot_gray_leaf_spot
Corn__common_rust
Corn__healthy
Corn__northern_leaf_blight
Grape__black_rot
Grape__esca_(black_measles)
Grape__healthy
Grape__leaf_blight_(isariopsis_leaf_spot)
Orange__haunglongbing_(citrus_greening)
Peach__bacterial_spot
Peach__healthy
Pepper_bell__bacterial_spot
Pepper_bell__healthy
Potato__early_blight
Potato__healthy
Potato__late_blight
Raspberry__healthy
Soybean__healthy
Squash__powdery_mildew
Strawberry__healthy
Strawberry__leaf_scorch
Tomato__bacterial_spot
Tomato__early_blight
Tomato__healthy
Tomato__late_blight
Tomato__leaf_mold
Tomato__mosaic_virus
Tomato__septoria_leaf_spot
Tomato__spider_mites_two_spotted_spider_mite
Tomato__target_spot
Tomato__yellow_leaf_curl_virus
```

Then run
```bash
python scripts/extract_small_dataset.py
```
to construct a low-resource dataset from the full PlantVillage data.

### CNN Baseline

To run baseline CNN training
```bash
python models/baseline_cnn/train.py
```

### CLIP

To run one-shot classification, please run
```bash
python models/clip/one_shot.py
```

To fine-tune CLIP end-to-end (text & image matching), please run
```bash
python models/clip/fine_tuning_end_to_end.py
```

To fine-tune CLIP with a classification head (cross entropy), please run
```bash
python models/clip/fine_tuning_classification.py
```

### BLIP

To run one-shot classification, please run
```bash
python models/blip/one_shot.py
```

To fine-tune BLIP end-to-end (text & image matching), please run
```bash
python models/blip/fine_tuning_end_to_end.py
```

To fine-tune BLIP with a classification head (cross entropy), please run (to be added)
```bash
python models/blip/fine_tuning_classification.py
```

### DINOv2

To run one-shot classification (k-nearest neighbors), please run
```bash
python models/dinov2/one_shot.py
```

To fine-tune DINOv2 with a classification head and simple data augmentation (rotate, flip, affine), please run
```bash
python models/dinov2/fine_tuning_classification.py
```

To fine-tune DINOv2 with a classification head and cut-mix augmentation, please run
```bash
python models/dinov2/fine_tuning_cut_mix.py
```

To fine-tune DINOv2 with a classification head and mix-up augmentation, please run
```bash
python models/dinov2/fine_tuning_mix_up.py
```


