"""
Author: Son Phat Tran
This file contains the utility code for labelling the diseases
"""

MAPPING_WITH_PREFIX = {
    "Apple__apple_scab": "An image of an apple leaf infected with apple scab disease",
    "Apple__black_rot": "An image of an apple leaf affected by black rot disease",
    "Apple__cedar_apple_rust": "An image of an apple leaf showing cedar apple rust infection",
    "Apple__healthy": "An image of a healthy apple tree leaf",
    "Background": "An image of a background or non-plant material",
    "Blueberry__healthy": "An image of a healthy blueberry plant leaf",
    "Cherry__healthy": "An image of a healthy cherry tree leaf",
    "Cherry__powdery_mildew": "An image of a cherry leaf infected with powdery mildew",
    "Corn__cercospora_leaf_spot_gray_leaf_spot": "An image of a corn leaf with gray leaf spot disease",
    "Corn__common_rust": "An image of a corn leaf infected with common rust",
    "Corn__healthy": "An image of a healthy corn plant leaf",
    "Corn__northern_leaf_blight": "An image of a corn leaf affected by northern leaf blight",
    "Grape__black_rot": "An image of a grape leaf with black rot disease",
    "Grape__esca_(black_measles)": "An image of a grape leaf showing esca disease symptoms",
    "Grape__healthy": "An image of a healthy grape vine leaf",
    "Grape__leaf_blight_(isariopsis_leaf_spot)": "An image of a grape leaf with isariopsis leaf spot disease",
    "Orange__haunglongbing_(citrus_greening)": "An image of an orange tree leaf affected by citrus greening disease",
    "Peach__bacterial_spot": "An image of a peach leaf with bacterial spot infection",
    "Peach__healthy": "An image of a healthy peach tree leaf",
    "Pepper_bell__bacterial_spot": "An image of a bell pepper leaf with bacterial spot disease",
    "Pepper_bell__healthy": "An image of a healthy bell pepper plant leaf",
    "Potato__early_blight": "An image of a potato leaf affected by early blight disease",
    "Potato__healthy": "An image of a healthy potato plant leaf",
    "Potato__late_blight": "An image of a potato leaf infected with late blight disease",
    "Raspberry__healthy": "An image of a healthy raspberry plant leaf",
    "Soybean__healthy": "An image of a healthy soybean plant leaf",
    "Squash__powdery_mildew": "An image of a squash leaf infected with powdery mildew",
    "Strawberry__healthy": "An image of a healthy strawberry plant leaf",
    "Strawberry__leaf_scorch": "An image of a strawberry leaf with leaf scorch disease",
    "Tomato__bacterial_spot": "An image of a tomato leaf with bacterial spot infection",
    "Tomato__early_blight": "An image of a tomato leaf affected by early blight disease",
    "Tomato__healthy": "An image of a healthy tomato plant leaf",
    "Tomato__late_blight": "An image of a tomato leaf infected with late blight disease",
    "Tomato__leaf_mold": "An image of a tomato leaf showing leaf mold infection",
    "Tomato__mosaic_virus": "An image of a tomato leaf infected with mosaic virus",
    "Tomato__septoria_leaf_spot": "An image of a tomato leaf with septoria leaf spot disease",
    "Tomato__spider_mites_two_spotted_spider_mite": "An image of a tomato leaf damaged by two-spotted spider mites",
    "Tomato__target_spot": "An image of a tomato leaf with target spot disease",
    "Tomato__yellow_leaf_curl_virus": "An image of a tomato leaf infected with yellow leaf curl virus"
}


MAPPING_WITHOUT_PREFIX = {
    "Apple__apple_scab": "An apple leaf infected with apple scab disease",
    "Apple__black_rot": "An apple leaf affected by black rot disease",
    "Apple__cedar_apple_rust": "An apple leaf showing cedar apple rust infection",
    "Apple__healthy": "A healthy apple tree leaf",
    "Background": "Background with non-plant material",
    "Blueberry__healthy": "A healthy blueberry plant leaf",
    "Cherry__healthy": "A healthy cherry tree leaf",
    "Cherry__powdery_mildew": "A cherry leaf infected with powdery mildew",
    "Corn__cercospora_leaf_spot_gray_leaf_spot": "A corn leaf with gray leaf spot disease",
    "Corn__common_rust": "A corn leaf infected with common rust",
    "Corn__healthy": "A healthy corn plant leaf",
    "Corn__northern_leaf_blight": "A corn leaf affected by northern leaf blight",
    "Grape__black_rot": "A grape leaf with black rot disease",
    "Grape__esca_(black_measles)": "A grape leaf showing esca disease symptoms",
    "Grape__healthy": "A healthy grape vine leaf",
    "Grape__leaf_blight_(isariopsis_leaf_spot)": "A grape leaf with isariopsis leaf spot disease",
    "Orange__haunglongbing_(citrus_greening)": "An orange tree leaf affected by citrus greening disease",
    "Peach__bacterial_spot": "A peach leaf with bacterial spot infection",
    "Peach__healthy": "A healthy peach tree leaf",
    "Pepper_bell__bacterial_spot": "A bell pepper leaf with bacterial spot disease",
    "Pepper_bell__healthy": "A healthy bell pepper plant leaf",
    "Potato__early_blight": "A potato leaf affected by early blight disease",
    "Potato__healthy": "A healthy potato plant leaf",
    "Potato__late_blight": "A potato leaf infected with late blight disease",
    "Raspberry__healthy": "A healthy raspberry plant leaf",
    "Soybean__healthy": "A healthy soybean plant leaf",
    "Squash__powdery_mildew": "A squash leaf infected with powdery mildew",
    "Strawberry__healthy": "A healthy strawberry plant leaf",
    "Strawberry__leaf_scorch": "A strawberry leaf with leaf scorch disease",
    "Tomato__bacterial_spot": "A tomato leaf with bacterial spot infection",
    "Tomato__early_blight": "A tomato leaf affected by early blight disease",
    "Tomato__healthy": "A healthy tomato plant leaf",
    "Tomato__late_blight": "A tomato leaf infected with late blight disease",
    "Tomato__leaf_mold": "A tomato leaf showing leaf mold infection",
    "Tomato__mosaic_virus": "A tomato leaf infected with mosaic virus",
    "Tomato__septoria_leaf_spot": "A tomato leaf with septoria leaf spot disease",
    "Tomato__spider_mites_two_spotted_spider_mite": "A tomato leaf damaged by two-spotted spider mites",
    "Tomato__target_spot": "A tomato leaf with target spot disease",
    "Tomato__yellow_leaf_curl_virus": "A tomato leaf infected with yellow leaf curl virus"
}


def convert_label(label: str, with_prefix=True) -> str:
    """
    Get the label for one-shot classification using image label
    :param with_prefix: map to label with prefix or not
    :param label: class label
    :return: CLIPv2 one-shot label
    """
    return MAPPING_WITH_PREFIX[label] if with_prefix else MAPPING_WITHOUT_PREFIX[label]
