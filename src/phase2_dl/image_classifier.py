"""
GeoSense Agent 2.0
Phase II - Exercise 5
Image Classification and NDVI-based Change Detection

Required functions:
    _load_model()
    _get_image_patch(lat, lon, radius_m)
    _compute_ndvi_ndwi(patch)
    classify_imagery(lat, lon, radius_m)

Required output fields:
    land_cover
    class_id
    confidence_pct
    ndvi
    ndwi
    ndvi_label
    class_distribution
    change_flag
    change_description
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import segmentation_models_pytorch as smp
from pyproj import Transformer


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "satellite" / "clean"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

CLEAN_2015 = DATA_DIR / "clean_2015.npy"
CLEAN_2023 = DATA_DIR / "clean_2023.npy"

PRITHVI_MODEL = MODEL_DIR / "prithvi_finetuned.pth"
UNET_MODEL = MODEL_DIR / "unet_best.pth"


# ============================================================
# 2. STUDY AREA AND DATA SETTINGS
# ============================================================

# AMC_EDIT.shp bounds used for the study area.
# CRS: EPSG:32646 (UTM Zone 46N)
# xmin, ymin, xmax, ymax

BBOX_UTM = (
    319623.8771,
    2630718.8950,
    331147.0646,
    2645370.2997,
)

SOURCE_CRS = "EPSG:4326"    # Latitude/Longitude
TARGET_CRS = "EPSG:32646"   # UTM Zone 46N

PATCH_SIZE = 224
INPUT_CHANNELS = 6
NUM_CLASSES = 5

# Lab requirement:
# Change = True when absolute NDVI difference > 0.15
CHANGE_THRESHOLD = 0.15


# ============================================================
# 3. SENTINEL-2 BAND ORDER
# ============================================================

# Six bands used in the Phase II preprocessing pipeline:
# B2, B3, B4, B8, B11, B12

B2 = 0
B3 = 1
B4 = 2
B8 = 3
B11 = 4
B12 = 5


# ============================================================
# 4. LAND-COVER CLASSES
# ============================================================

CLASS_NAMES = {
    0: "Urban",
    1: "Vegetation",
    2: "Water",
    3: "Bare Land",
    4: "Agriculture",
}


# ============================================================
# 5. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 6. COORDINATE TRANSFORMER
# ============================================================

TRANSFORMER = Transformer.from_crs(
    SOURCE_CRS,
    TARGET_CRS,
    always_xy=True
)


# ============================================================
# 7. GLOBAL VARIABLES
# ============================================================

IMAGE_2015 = None
IMAGE_2023 = None

MODEL = None
MODEL_USED = None

DATA_INITIALIZED = False


# ============================================================
# 8. INITIALIZE PREPROCESSED DATA
# ============================================================

def _initialize_data():
    """
    Load preprocessed 2015 and 2023 six-band imagery.
    """

    global IMAGE_2015
    global IMAGE_2023
    global DATA_INITIALIZED

    if DATA_INITIALIZED:
        return

    # Check files
    if not CLEAN_2015.exists():
        raise FileNotFoundError(
            f"2015 imagery not found:\n{CLEAN_2015}"
        )

    if not CLEAN_2023.exists():
        raise FileNotFoundError(
            f"2023 imagery not found:\n{CLEAN_2023}"
        )

    # Load arrays
    IMAGE_2015 = np.load(
        CLEAN_2015,
        mmap_mode="r"
    )

    IMAGE_2023 = np.load(
        CLEAN_2023,
        mmap_mode="r"
    )

    # Validate dimensions
    if IMAGE_2015.ndim != 3:
        raise ValueError(
            "2015 image must have shape (bands, height, width)."
        )

    if IMAGE_2023.ndim != 3:
        raise ValueError(
            "2023 image must have shape (bands, height, width)."
        )

    if IMAGE_2015.shape[0] != INPUT_CHANNELS:
        raise ValueError(
            f"2015 image has {IMAGE_2015.shape[0]} bands; "
            f"expected {INPUT_CHANNELS}."
        )

    if IMAGE_2023.shape[0] != INPUT_CHANNELS:
        raise ValueError(
            f"2023 image has {IMAGE_2023.shape[0]} bands; "
            f"expected {INPUT_CHANNELS}."
        )

    if IMAGE_2015.shape[1:] != IMAGE_2023.shape[1:]:
        raise ValueError(
            "2015 and 2023 images do not have the same "
            f"spatial dimensions:\n"
            f"2015 = {IMAGE_2015.shape}\n"
            f"2023 = {IMAGE_2023.shape}"
        )

    DATA_INITIALIZED = True

    print("Preprocessed imagery loaded successfully.")
    print(f"2015 shape: {IMAGE_2015.shape}")
    print(f"2023 shape: {IMAGE_2023.shape}")


# ============================================================
# 9. EXTRACT STATE DICTIONARY FROM MODEL CHECKPOINT
# ============================================================

def _extract_state_dict(checkpoint):
    """
    Handle common PyTorch checkpoint formats.
    """

    # Case 1: direct state dictionary
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

        elif "model" in checkpoint and isinstance(
            checkpoint["model"], dict
        ):
            checkpoint = checkpoint["model"]

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Invalid model checkpoint format."
        )

    cleaned_state_dict = {}

    for key, value in checkpoint.items():

        # Remove DataParallel prefix
        if key.startswith("module."):
            key = key.replace(
                "module.",
                "",
                1
            )

        # Remove model prefix if present
        if key.startswith("model."):
            key = key.replace(
                "model.",
                "",
                1
            )

        cleaned_state_dict[key] = value

    return cleaned_state_dict


# ============================================================
# 10. LOAD MODEL
# ============================================================

def _load_model():
    """
    Load the fine-tuned model first.

    Preferred:
        prithvi_finetuned.pth

    Fallback:
        unet_best.pth
    """

    global MODEL
    global MODEL_USED

    if MODEL is not None:
        return MODEL, MODEL_USED

    # --------------------------------------------------------
    # TRY FINE-TUNED RESNET-50 MODEL FIRST
    # --------------------------------------------------------

    if PRITHVI_MODEL.exists():

        try:

            print("\nTrying fine-tuned model...")
            print(PRITHVI_MODEL)

            model = smp.Unet(
                encoder_name="resnet50",
                encoder_weights=None,
                in_channels=INPUT_CHANNELS,
                classes=NUM_CLASSES,
                activation=None,
            )

            checkpoint = torch.load(
                PRITHVI_MODEL,
                map_location=DEVICE,
                weights_only=False
            )

            state_dict = _extract_state_dict(
                checkpoint
            )

            result = model.load_state_dict(
                state_dict,
                strict=False
            )

            if len(result.missing_keys) > 0:

                raise RuntimeError(
                    "Fine-tuned checkpoint has missing "
                    f"keys: {len(result.missing_keys)}"
                )

            model = model.to(DEVICE)
            model.eval()

            MODEL = model
            MODEL_USED = (
                "ResNet-50 Fine-tuned Substitute"
            )

            print(
                "Fine-tuned model loaded successfully."
            )

            print(
                f"Model: {MODEL_USED}"
            )

            print(
                f"Device: {DEVICE}"
            )

            return MODEL, MODEL_USED

        except Exception as error:

            print(
                "\nWarning: fine-tuned model could not "
                "be loaded."
            )

            print(
                f"Reason: {error}"
            )

            print(
                "Falling back to ResNet-34 U-Net..."
            )

    # --------------------------------------------------------
    # FALLBACK TO EXERCISE 3 U-NET
    # --------------------------------------------------------

    if UNET_MODEL.exists():

        print("\nLoading fallback U-Net model...")
        print(UNET_MODEL)

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=INPUT_CHANNELS,
            classes=NUM_CLASSES,
            activation=None,
        )

        checkpoint = torch.load(
            UNET_MODEL,
            map_location=DEVICE,
            weights_only=False
        )

        state_dict = _extract_state_dict(
            checkpoint
        )

        result = model.load_state_dict(
            state_dict,
            strict=False
        )

        if len(result.missing_keys) > 0:

            raise RuntimeError(
                "Fallback U-Net checkpoint has missing "
                f"keys: {len(result.missing_keys)}"
            )

        model = model.to(DEVICE)
        model.eval()

        MODEL = model
        MODEL_USED = "ResNet-34 U-Net"

        print(
            "Fallback U-Net loaded successfully."
        )

        print(
            f"Model: {MODEL_USED}"
        )

        print(
            f"Device: {DEVICE}"
        )

        return MODEL, MODEL_USED

    # --------------------------------------------------------
    # NO MODEL AVAILABLE
    # --------------------------------------------------------

    raise FileNotFoundError(
        "\nNo trained model was found.\n"
        f"Expected:\n"
        f"1. {PRITHVI_MODEL}\n"
        f"2. {UNET_MODEL}"
    )


# ============================================================
# 11. LAT/LON TO PIXEL COORDINATES
# ============================================================

def _latlon_to_pixel(
    lat: float,
    lon: float
) -> Tuple[int, int]:
    """
    Convert latitude/longitude into raster row/column
    using the study-area BBOX.
    """

    _initialize_data()

    easting, northing = TRANSFORMER.transform(
        lon,
        lat
    )

    xmin, ymin, xmax, ymax = BBOX_UTM

    # Check whether the point is inside the study area
    if not (
        xmin <= easting <= xmax
        and
        ymin <= northing <= ymax
    ):

        raise ValueError(
            f"Location ({lat}, {lon}) is outside "
            "the AMC study-area BBOX."
        )

    height = IMAGE_2023.shape[1]
    width = IMAGE_2023.shape[2]

    # Convert coordinates to fractional pixel position
    x_fraction = (
        (easting - xmin)
        /
        (xmax - xmin)
    )

    y_fraction = (
        (ymax - northing)
        /
        (ymax - ymin)
    )

    col = int(
        round(
            x_fraction * (width - 1)
        )
    )

    row = int(
        round(
            y_fraction * (height - 1)
        )
    )

    # Keep inside raster boundaries
    row = max(
        0,
        min(row, height - 1)
    )

    col = max(
        0,
        min(col, width - 1)
    )

    return row, col


# ============================================================
# 12. EXTRACT IMAGE PATCH
# ============================================================

def _get_image_patch(
    lat: float,
    lon: float,
    radius_m: float = 500.0
):
    """
    Extract corresponding 224 x 224 x 6 patches
    for 2015 and 2023.

    The lab requires a fixed 224 x 224 patch.
    """

    _initialize_data()

    row, col = _latlon_to_pixel(
        lat,
        lon
    )

    height = IMAGE_2023.shape[1]
    width = IMAGE_2023.shape[2]

    half = PATCH_SIZE // 2

    # Initial window
    row_start = row - half
    col_start = col - half

    row_end = row_start + PATCH_SIZE
    col_end = col_start + PATCH_SIZE

    # Shift window when it reaches raster edges
    if row_start < 0:

        row_start = 0
        row_end = PATCH_SIZE

    if col_start < 0:

        col_start = 0
        col_end = PATCH_SIZE

    if row_end > height:

        row_end = height
        row_start = height - PATCH_SIZE

    if col_end > width:

        col_end = width
        col_start = width - PATCH_SIZE

    # Final validation
    if (
        row_start < 0
        or
        col_start < 0
        or
        row_end > height
        or
        col_end > width
    ):

        raise ValueError(
            "Unable to extract a 224 x 224 patch "
            "from the image."
        )

    # Extract 2015 and 2023 patches
    patch_2015 = np.asarray(
        IMAGE_2015[
            :,
            row_start:row_end,
            col_start:col_end
        ],
        dtype=np.float32
    ).copy()

    patch_2023 = np.asarray(
        IMAGE_2023[
            :,
            row_start:row_end,
            col_start:col_end
        ],
        dtype=np.float32
    ).copy()

    # --------------------------------------------------------
    # Fill NaN / infinite values using band mean
    # --------------------------------------------------------

    for patch in [
        patch_2015,
        patch_2023
    ]:

        for band in range(INPUT_CHANNELS):

            band_data = patch[band]

            valid = np.isfinite(
                band_data
            )

            if np.any(valid):

                band_mean = float(
                    np.mean(
                        band_data[valid]
                    )
                )

            else:

                band_mean = 0.0

            band_data[~valid] = band_mean

    return (
        patch_2015,
        patch_2023,
        row,
        col
    )


# ============================================================
# 13. SAFE NORMALIZED DIFFERENCE
# ============================================================

def _normalized_difference(
    numerator_band: np.ndarray,
    denominator_band: np.ndarray
) -> np.ndarray:
    """
    Calculate:
        (A - B) / (A + B)

    with division-by-zero protection.
    """

    denominator = (
        numerator_band
        +
        denominator_band
    )

    result = np.zeros_like(
        numerator_band,
        dtype=np.float32
    )

    valid = (
        np.abs(denominator)
        >
        1e-8
    )

    result[valid] = (
        (
            numerator_band[valid]
            -
            denominator_band[valid]
        )
        /
        denominator[valid]
    )

    return np.clip(
        result,
        -1.0,
        1.0
    )


# ============================================================
# 14. NDVI AND NDWI
# ============================================================

def _compute_ndvi_ndwi(
    patch: np.ndarray
):
    """
    Compute NDVI and NDWI.

    NDVI = (NIR - Red) / (NIR + Red)
    NDWI = (Green - NIR) / (Green + NIR)
    """

    if patch.shape[0] != INPUT_CHANNELS:

        raise ValueError(
            f"Expected {INPUT_CHANNELS} bands, "
            f"received {patch.shape[0]}."
        )

    red = patch[B4]
    nir = patch[B8]
    green = patch[B3]

    ndvi = _normalized_difference(
        nir,
        red
    )

    ndwi = _normalized_difference(
        green,
        nir
    )

    return ndvi, ndwi


# ============================================================
# 15. NDVI INTERPRETATION
# ============================================================

def _get_ndvi_label(
    ndvi: float
) -> str:
    """
    Give a simple interpretation of mean NDVI.
    """

    if ndvi < 0.0:

        return "Non-vegetated / Water"

    elif ndvi < 0.20:

        return "Low vegetation"

    elif ndvi < 0.40:

        return "Moderate vegetation"

    elif ndvi < 0.60:

        return "High vegetation"

    else:

        return "Very high vegetation"


# ============================================================
# 16. MODEL INFERENCE
# ============================================================

def _predict_patch(
    patch: np.ndarray
):
    """
    Perform segmentation inference.

    Returns:
        predicted class mask
        average confidence
        inference time
    """

    model, _ = _load_model()

    tensor = torch.from_numpy(
        patch
    ).float().unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------
    # Warm-up is not included in reported inference time.
    with torch.inference_mode():

        _ = model(tensor)

    # --------------------------------------------------------
    # Timed inference
    # --------------------------------------------------------

    start_time = time.perf_counter()

    with torch.inference_mode():

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence_map, prediction = torch.max(
            probabilities,
            dim=1
        )

    inference_time = (
        time.perf_counter()
        -
        start_time
    )

    prediction = (
        prediction
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    confidence = float(
        confidence_map
        .mean()
        .item()
        *
        100.0
    )

    return (
        prediction,
        confidence,
        inference_time
    )


# ============================================================
# 17. CLASS DISTRIBUTION
# ============================================================

def _get_class_distribution(
    prediction: np.ndarray
) -> Dict[str, float]:
    """
    Calculate percentage of each predicted
    land-cover class in the patch.
    """

    total_pixels = prediction.size

    distribution = {}

    for class_id, class_name in CLASS_NAMES.items():

        count = int(
            np.count_nonzero(
                prediction == class_id
            )
        )

        percentage = (
            count
            /
            total_pixels
            *
            100.0
        )

        distribution[class_name] = round(
            percentage,
            2
        )

    return distribution


# ============================================================
# 18. MAIN CLASSIFICATION FUNCTION
# ============================================================

def classify_imagery(
    lat: float,
    lon: float,
    radius_m: float = 500.0
) -> Dict[str, object]:
    """
    Classify a location using 2023 imagery and
    compare the corresponding location with 2015.

    Returns all fields required by Exercise 5.
    """

    # --------------------------------------------------------
    # Extract matching patches
    # --------------------------------------------------------

    (
        patch_2015,
        patch_2023,
        center_row,
        center_col
    ) = _get_image_patch(
        lat,
        lon,
        radius_m
    )

    # --------------------------------------------------------
    # Calculate indices
    # --------------------------------------------------------

    ndvi_2015, _ = _compute_ndvi_ndwi(
        patch_2015
    )

    ndvi_2023, ndwi_2023 = _compute_ndvi_ndwi(
        patch_2023
    )

    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------

    (
        prediction,
        confidence_pct,
        inference_time
    ) = _predict_patch(
        patch_2023
    )

    # --------------------------------------------------------
    # Patch-level NDVI / NDWI
    # --------------------------------------------------------

    mean_ndvi_2023 = float(
        np.mean(ndvi_2023)
    )

    mean_ndwi_2023 = float(
        np.mean(ndwi_2023)
    )

    # --------------------------------------------------------
    # Center-pixel NDVI change
    # --------------------------------------------------------
    #
    # The selected latitude/longitude is the test location.
    # Therefore the change flag compares the corresponding
    # center pixel in 2015 and 2023.
    #
    # Required rule:
    # abs(NDVI_2023 - NDVI_2015) > 0.15
    # --------------------------------------------------------

    center = PATCH_SIZE // 2

    center_ndvi_2015 = float(
        ndvi_2015[
            center,
            center
        ]
    )

    center_ndvi_2023 = float(
        ndvi_2023[
            center,
            center
        ]
    )

    ndvi_difference = (
        center_ndvi_2023
        -
        center_ndvi_2015
    )

    absolute_ndvi_difference = abs(
        ndvi_difference
    )

    change_flag = (
        absolute_ndvi_difference
        >
        CHANGE_THRESHOLD
    )

    # --------------------------------------------------------
    # Percentage of changed pixels in the complete patch
    # This is descriptive information only.
    # --------------------------------------------------------

    changed_pixels = (
        np.abs(
            ndvi_2023
            -
            ndvi_2015
        )
        >
        CHANGE_THRESHOLD
    )

    changed_pixel_percentage = float(
        np.mean(
            changed_pixels
        )
        *
        100.0
    )

    # --------------------------------------------------------
    # Majority predicted class
    # --------------------------------------------------------

    class_counts = np.bincount(
        prediction.ravel(),
        minlength=NUM_CLASSES
    )

    class_id = int(
        np.argmax(
            class_counts
        )
    )

    land_cover = CLASS_NAMES[
        class_id
    ]

    # --------------------------------------------------------
    # NDVI interpretation
    # --------------------------------------------------------

    ndvi_label = _get_ndvi_label(
        mean_ndvi_2023
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_distribution = (
        _get_class_distribution(
            prediction
        )
    )

    # --------------------------------------------------------
    # Change description
    # --------------------------------------------------------

    if change_flag:

        if ndvi_difference > 0:
            direction = "increase"
        else:
            direction = "decrease"

        change_description = (
            f"Significant NDVI change detected at "
            f"the selected location. NDVI shows a "
            f"{direction} of "
            f"{absolute_ndvi_difference:.4f}, "
            f"which is greater than the threshold "
            f"of {CHANGE_THRESHOLD:.2f}."
        )

    else:

        change_description = (
            f"No significant NDVI change detected "
            f"at the selected location. Absolute "
            f"NDVI difference is "
            f"{absolute_ndvi_difference:.4f}, "
            f"which is within the threshold of "
            f"{CHANGE_THRESHOLD:.2f}."
        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    result = {

        # Required Exercise 5 fields
        "land_cover": land_cover,

        "class_id": class_id,

        "confidence_pct": round(
            confidence_pct,
            2
        ),

        "ndvi": round(
            mean_ndvi_2023,
            4
        ),

        "ndwi": round(
            mean_ndwi_2023,
            4
        ),

        "ndvi_label": ndvi_label,

        "class_distribution":
            class_distribution,

        "change_flag":
            bool(change_flag),

        "change_description":
            change_description,

        # Supporting information
        "ndvi_2015": round(
            center_ndvi_2015,
            4
        ),

        "ndvi_2023_center": round(
            center_ndvi_2023,
            4
        ),

        "ndvi_difference": round(
            ndvi_difference,
            4
        ),

        "absolute_ndvi_difference":
            round(
                absolute_ndvi_difference,
                4
            ),

        "changed_pixel_percentage":
            round(
                changed_pixel_percentage,
                2
            ),

        "model_used":
            MODEL_USED,

        "latitude":
            lat,

        "longitude":
            lon,

        "radius_m":
            radius_m,

        "patch_size":
            "224 x 224",

        "input_channels":
            INPUT_CHANNELS,

        "device":
            str(DEVICE),

        "center_pixel":
            {
                "row": int(center_row),
                "column": int(center_col)
            },

        "inference_time_sec":
            round(
                inference_time,
                4
            ),
    }

    return result


# ============================================================
# 19. PRINT RESULTS
# ============================================================

def print_result(
    result: Dict[str, object]
):
    """
    Print a clean output suitable for screenshots
    in the practical record.
    """

    print()
    print("=" * 70)
    print(
        "GeoSense Agent 2.0 - "
        "Phase II Exercise 5"
    )
    print("=" * 70)

    print(
        f"Coordinates        : "
        f"({result['latitude']}, "
        f"{result['longitude']})"
    )

    print(
        f"Model used         : "
        f"{result['model_used']}"
    )

    print(
        f"Patch              : "
        f"{result['patch_size']} "
        f"with {result['input_channels']} bands"
    )

    print(
        f"Land cover         : "
        f"{result['land_cover']}"
    )

    print(
        f"Class ID           : "
        f"{result['class_id']}"
    )

    print(
        f"Confidence         : "
        f"{result['confidence_pct']} %"
    )

    print(
        f"NDVI (2023 patch)  : "
        f"{result['ndvi']}"
    )

    print(
        f"NDWI (2023 patch)  : "
        f"{result['ndwi']}"
    )

    print(
        f"NDVI label         : "
        f"{result['ndvi_label']}"
    )

    print(
        f"NDVI 2015 center   : "
        f"{result['ndvi_2015']}"
    )

    print(
        f"NDVI 2023 center   : "
        f"{result['ndvi_2023_center']}"
    )

    print(
        f"NDVI difference    : "
        f"{result['ndvi_difference']}"
    )

    print(
        f"Absolute difference: "
        f"{result['absolute_ndvi_difference']}"
    )

    print(
        f"Changed pixels     : "
        f"{result['changed_pixel_percentage']} %"
    )

    print(
        f"Change flag        : "
        f"{result['change_flag']}"
    )

    print(
        f"Change description : "
        f"{result['change_description']}"
    )

    print(
        "Class distribution : "
        + json.dumps(
            result["class_distribution"],
            indent=None
        )
    )

    print(
        f"Inference time     : "
        f"{result['inference_time_sec']} seconds"
    )

    print("=" * 70)


# ============================================================
# 20. COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "GeoSense Agent 2.0 "
            "Phase II Exercise 5 classifier"
        )
    )

    # Changed location from the actual test
    parser.add_argument(
        "--lat",
        type=float,
        default=23.825971
    )

    parser.add_argument(
        "--lon",
        type=float,
        default=91.268249
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=500.0
    )

    args = parser.parse_args()

    result = classify_imagery(
        lat=args.lat,
        lon=args.lon,
        radius_m=args.radius
    )

    print_result(result)


# ============================================================
# 21. RUN
# ============================================================

if __name__ == "__main__":
    main()