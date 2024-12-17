import os
import difflib
import glob

NAME = [
    "Apple__apple_scab",
    "Apple__black_rot",
    "Apple__cedar_apple_rust",
    "Apple__healthy",
    "Background",
    "Blueberry__healthy",
    "Cherry__healthy",
    "Cherry__powdery_mildew",
    "Corn__cercospora_leaf_spot_gray_leaf_spot",
    "Corn__common_rust",
    "Corn__healthy",
    "Corn__northern_leaf_blight",
    "Grape__black_rot",
    "Grape__esca_(black_measles)",
    "Grape__healthy",
    "Grape__leaf_blight_(isariopsis_leaf_spot)",
    "Orange__haunglongbing_(citrus_greening)",
    "Peach__bacterial_spot",
    "Peach__healthy",
    "Pepper_bell__bacterial_spot",
    "Pepper_bell__healthy",
    "Potato__early_blight",
    "Potato__healthy",
    "Potato__late_blight",
    "Raspberry__healthy",
    "Soybean__healthy",
    "Squash__powdery_mildew",
    "Strawberry__healthy",
    "Strawberry__leaf_scorch",
    "Tomato__bacterial_spot",
    "Tomato__early_blight",
    "Tomato__healthy",
    "Tomato__late_blight",
    "Tomato__leaf_mold",
    "Tomato__mosaic_virus",
    "Tomato__septoria_leaf_spot",
    "Tomato__spider_mites_two_spotted_spider_mite",
    "Tomato__target_spot",
    "Tomato__yellow_leaf_curl_virus",
    "Background"
]

DATA_PATH = "raw_dataset/full" 

for folder in os.listdir(DATA_PATH):
    folder_name = os.path.basename(folder)
    closest_match = difflib.get_close_matches(folder_name, NAME, n=1)
    NAME.pop(NAME.index(closest_match[0]))
    if closest_match:
        print(f"Change name from {folder_name} -> {closest_match[0]}")
        os.rename(os.path.join(DATA_PATH, folder), os.path.join(DATA_PATH, closest_match[0]))
    else:
        raise ValueError(f"Folder {folder_name} not found in NAME")
