import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------
# GeoSense Phase II - Exercise 3.1
# Generate rule-based pseudo-labels
# ---------------------------------------------------------

CLEAN_DIR = Path("data/satellite/clean")
LABEL_DIR = Path("data/satellite/labels")

LABEL_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2015, 2023]

# Band order:
# B2, B3, B4, B8, B11, B12
B8 = 3
B11 = 4


# Class IDs
URBAN = 0
VEGETATION = 1
WATER = 2
BARE_LAND = 3
AGRICULTURE = 4

CLASS_NAMES = {
    URBAN: "Urban",
    VEGETATION: "Vegetation",
    WATER: "Water",
    BARE_LAND: "Bare Land",
    AGRICULTURE: "Agriculture"
}


def classify_pixel(ndvi, ndwi, nir, swir):
    """
    Apply the rule-based classification thresholds
    specified in the Phase II laboratory manual.

    Rules:
        NDWI > 0.3              -> Water
        NDVI > 0.4              -> Vegetation
        NDVI > 0.15             -> Agriculture
        NDVI < 0.05 & SWIR > 0.2 -> Urban
        Otherwise               -> Bare Land
    """

    if ndwi > 0.3:
        return WATER

    elif ndvi > 0.4:
        return VEGETATION

    elif ndvi > 0.15:
        return AGRICULTURE

    elif ndvi < 0.05 and swir > 0.2:
        return URBAN

    else:
        return BARE_LAND


def create_label_mask(data, ndvi, ndwi):
    """
    Create a complete land-cover label mask
    from the spectral rules.
    """

    nir = data[B8]
    swir = data[B11]

    labels = np.full(
        ndvi.shape,
        BARE_LAND,
        dtype=np.uint8
    )

    # Rule 1: Water
    water_mask = ndwi > 0.3
    labels[water_mask] = WATER

    # Rule 2: Vegetation
    vegetation_mask = (
        (~water_mask) &
        (ndvi > 0.4)
    )
    labels[vegetation_mask] = VEGETATION

    # Rule 3: Agriculture
    agriculture_mask = (
        (~water_mask) &
        (~vegetation_mask) &
        (ndvi > 0.15)
    )
    labels[agriculture_mask] = AGRICULTURE

    # Rule 4: Urban
    urban_mask = (
        (~water_mask) &
        (~vegetation_mask) &
        (~agriculture_mask) &
        (ndvi < 0.05) &
        (swir > 0.2)
    )
    labels[urban_mask] = URBAN

    return labels


def calculate_class_distribution(labels):
    """
    Calculate class counts and percentages.
    """

    total_pixels = labels.size

    distribution = {}

    for class_id, class_name in CLASS_NAMES.items():

        count = int(np.sum(labels == class_id))

        percentage = (
            count / total_pixels
        ) * 100.0

        distribution[class_name] = {
            "class_id": class_id,
            "pixel_count": count,
            "percentage": percentage
        }

    return distribution


def process_year(year):

    clean_file = CLEAN_DIR / f"clean_{year}.npy"
    ndvi_file = CLEAN_DIR / f"ndvi_{year}.npy"
    ndwi_file = CLEAN_DIR / f"ndwi_{year}.npy"

    if not clean_file.exists():
        raise FileNotFoundError(
            f"Clean file not found: {clean_file}"
        )

    if not ndvi_file.exists():
        raise FileNotFoundError(
            f"NDVI file not found: {ndvi_file}"
        )

    if not ndwi_file.exists():
        raise FileNotFoundError(
            f"NDWI file not found: {ndwi_file}"
        )

    print("\n" + "=" * 60)
    print(f"Generating pseudo-labels for {year}")
    print("=" * 60)

    data = np.load(clean_file)
    ndvi = np.load(ndvi_file)
    ndwi = np.load(ndwi_file)

    print("Clean data shape:", data.shape)
    print("NDVI shape:", ndvi.shape)
    print("NDWI shape:", ndwi.shape)

    labels = create_label_mask(
        data,
        ndvi,
        ndwi
    )

    distribution = calculate_class_distribution(
        labels
    )

    # Save label mask
    label_file = (
        LABEL_DIR / f"labels_{year}.npy"
    )

    np.save(
        label_file,
        labels
    )

    # Save distribution
    distribution_file = (
        LABEL_DIR /
        f"class_distribution_{year}.json"
    )

    metadata = {
        "year": year,
        "classes": CLASS_NAMES,
        "distribution": distribution
    }

    with open(
        distribution_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    print("\nClass distribution:")

    for class_name, values in distribution.items():

        print(
            f"{class_name}: "
            f"{values['pixel_count']} pixels "
            f"({values['percentage']:.2f}%)"
        )

    print("\nSaved files:")
    print(label_file)
    print(distribution_file)


def main():

    print(
        "GeoSense Phase II - "
        "Rule-based Pseudo-labelling"
    )

    print("Years:", YEARS)

    for year in YEARS:
        process_year(year)

    print("\n" + "=" * 60)
    print(
        "Pseudo-labelling completed successfully."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()