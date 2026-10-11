from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

# ==========================================================
# GEOSENSE PHASE III - EXERCISE 4
# Street-Level Semantic Segmentation & Walkability Scoring
# ==========================================================

PROJECT_DIR = Path(r"D:\GeoSense_Phase3")
IMAGE_DIR = PROJECT_DIR / "data" / "streetview" / "images"
RESULTS_DIR = PROJECT_DIR / "data" / "streetview" / "results"
METADATA_CSV = RESULTS_DIR / "mapillary_image_metadata.csv"

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

CLASS_NAMES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    8: "vegetation",
    10: "sky",
    13: "car",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = None
model = None

BASELINE_WEIGHTS = {"sidewalk": 0.4, "vegetation": 0.3, "car": -0.2}
ALTERNATIVE_WEIGHTS = {"sidewalk": 0.5, "vegetation": 0.2, "car": -0.3}

# Coordinates recovered from the original metadata that was previously
# downloaded. The existing image files are not changed or deleted.
# The image URL is intentionally blank for these restored rows.
PREVIOUS_IMAGE_COORDINATES = {
    "3923803637702041.jpg": (13.051388, 80.244341),
    "1460715957610905.jpg": (13.051447, 80.244187),
    "540995766883232.jpg": (13.050865, 80.245619),
    "468631777526818.jpg": (13.051477, 80.244104),
    "469482657493384.jpg": (13.059966, 80.242714),
    "128979309225141.jpg": (13.050934, 80.245599),
    "1445591792446080.jpg": (13.050901, 80.245618),
    "450748069344852.jpg": (13.050564, 80.245588),
    "4633727689977364.jpg": (13.051307, 80.244595),
    "188772236399983.jpg": (13.051604, 80.243778),
    "298844998364039.jpg": (13.050965, 80.245542),
    "1022032611891437.jpg": (13.051735, 80.243417),
    "1061380500937981.jpg": (13.051866, 80.243086),
    "112970120832999.jpg": (13.050708, 80.245613),
    "460391951833317.jpg": (13.050206, 80.245544),
}


def load_model():
    global processor, model
    if processor is None or model is None:
        print("Loading model:", MODEL_NAME)
        processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        model.eval()
        print("Model loaded on:", DEVICE)
    return processor, model


def classify_image(image_path):
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    image_processor, segmentation_model = load_model()
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = segmentation_model(**inputs)

    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )
    prediction = logits.argmax(dim=1)[0].cpu().numpy()
    total_pixels = prediction.size

    composition = {}
    for class_id, class_name in CLASS_NAMES.items():
        count = np.count_nonzero(prediction == class_id)
        composition[class_name] = round(100.0 * count / total_pixels, 1)
    return composition


def calculate_walkability(composition, weights=None):
    """Practicum formula; output is clipped to the range 0-10."""
    if weights is None:
        weights = BASELINE_WEIGHTS
    raw_score = (
        composition.get("sidewalk", 0.0) * weights["sidewalk"]
        + composition.get("vegetation", 0.0) * weights["vegetation"]
        + composition.get("car", 0.0) * weights["car"]
    ) / 5.0
    return round(float(np.clip(raw_score, 0, 10)), 2)


def load_metadata():
    """Load CSV and restore coordinate rows for the original 15 images."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if METADATA_CSV.exists():
        df = pd.read_csv(METADATA_CSV, dtype={"id": str})
    else:
        df = pd.DataFrame()

    # Ensure all required columns exist, even if an older CSV has fewer.
    required_columns = [
        "id", "filename", "latitude", "longitude",
        "thumb_2048_url", "compass_angle", "captured_at", "area",
    ]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    # Restore metadata for original files if their records were lost.
    existing_filenames = set(df["filename"].astype(str))
    restored = []
    for filename, (lat, lon) in PREVIOUS_IMAGE_COORDINATES.items():
        if filename not in existing_filenames:
            image_id = Path(filename).stem
            restored.append({
                "id": image_id,
                "filename": filename,
                "latitude": lat,
                "longitude": lon,
                "thumb_2048_url": "",
                "compass_angle": "",
                "captured_at": "",
                "area": "Original study area (restored coordinates)",
            })

    if restored:
        df = pd.concat([df, pd.DataFrame(restored)], ignore_index=True)
        print("Restored original image coordinate records:", len(restored))

    # Add a row for any image file not listed in the CSV, but do not
    # invent coordinates. Such rows can be classified but not spatially queried.
    listed = set(df["filename"].dropna().astype(str))
    for path in sorted(IMAGE_DIR.glob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if path.name not in listed:
                df.loc[len(df)] = {
                    "id": path.stem,
                    "filename": path.name,
                    "latitude": np.nan,
                    "longitude": np.nan,
                    "thumb_2048_url": "",
                    "compass_angle": "",
                    "captured_at": "",
                    "area": "Image file; coordinates unavailable",
                }

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.drop_duplicates(subset=["filename"], keep="last")
    df.to_csv(METADATA_CSV, index=False, encoding="utf-8-sig")
    return df


def analyze_location(lat, lon, compositions_by_filename, metadata, radius_deg=0.0015):
    """Average classified imagery in an approximate lat/lon bounding box."""
    subset = metadata[
        metadata["latitude"].between(lat - radius_deg, lat + radius_deg)
        & metadata["longitude"].between(lon - radius_deg, lon + radius_deg)
    ]

    image_compositions = []
    filenames_used = []

    for _, row in subset.iterrows():
        filename = str(row["filename"])
        if filename in compositions_by_filename:
            image_compositions.append(compositions_by_filename[filename])
            filenames_used.append(filename)

    if not image_compositions:
        return {
            "latitude": lat,
            "longitude": lon,
            "images_analyzed": 0,
            "scene_composition": None,
            "walkability_score": None,
            "images_used": [],
            "error": "No classified imagery within this coordinate box.",
        }

    mean_composition = {
        name: round(float(np.mean([item[name] for item in image_compositions])), 1)
        for name in CLASS_NAMES.values()
    }

    return {
        "latitude": lat,
        "longitude": lon,
        "images_analyzed": len(image_compositions),
        "scene_composition": mean_composition,
        "walkability_score": calculate_walkability(mean_composition),
        "images_used": filenames_used,
        "error": None,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    metadata = load_metadata()
    image_files = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    print("=" * 60)
    print("GEOSENSE PHASE III - EXERCISE 4")
    print("=" * 60)
    print("Image files found:", len(image_files))
    print("Metadata rows:", len(metadata))
    print("Rows with coordinates:", int(metadata[["latitude", "longitude"]].notna().all(axis=1).sum()))
    print("Processing device:", DEVICE)

    compositions_by_filename = {}
    image_rows = []

    for image_path in image_files:
        try:
            composition = classify_image(image_path)
            compositions_by_filename[image_path.name] = composition
            row = {
                "filename": image_path.name,
                **composition,
                "baseline_walkability_score": calculate_walkability(
                    composition, BASELINE_WEIGHTS
                ),
                "alternative_walkability_score": calculate_walkability(
                    composition, ALTERNATIVE_WEIGHTS
                ),
            }
            image_rows.append(row)
            print(
                f"{image_path.name} | sidewalk={composition['sidewalk']}% | "
                f"vegetation={composition['vegetation']}% | car={composition['car']}% | "
                f"baseline={row['baseline_walkability_score']}"
            )
        except Exception as exc:
            print(f"Failed to process {image_path.name}: {type(exc).__name__}: {exc}")

    if not image_rows:
        print("No images were successfully classified.")
        return

    image_df = pd.DataFrame(image_rows)
    image_csv = RESULTS_DIR / "streetview_semantic_results.csv"
    image_df.to_csv(image_csv, index=False, encoding="utf-8-sig")
    print("\nImage results saved:", image_csv)

    # Required Chennai centre plus two locations in the original downloaded cluster.
    locations = [
        ("Chennai_Centre", 13.0827, 80.2707),
        ("Original_Area_1", 13.0508, 80.2456),
        ("Original_Area_2", 13.0515, 80.2442),
    ]

    location_results = []
    for location_id, lat, lon in locations:
        print(f"\nAnalyzing {location_id}: {lat}, {lon}")
        result = analyze_location(lat, lon, compositions_by_filename, metadata)
        result["location_id"] = location_id
        location_results.append(result)
        print(json.dumps(result, indent=2))

    location_json = RESULTS_DIR / "streetview_location_results.json"
    location_json.write_text(json.dumps(location_results, indent=2), encoding="utf-8")

    location_rows = []
    for result in location_results:
        row = {
            "location_id": result["location_id"],
            "latitude": result["latitude"],
            "longitude": result["longitude"],
            "images_analyzed": result["images_analyzed"],
            "walkability_score": result["walkability_score"],
            "error": result["error"],
        }
        if result["scene_composition"]:
            row.update(result["scene_composition"])
        location_rows.append(row)

    location_csv = RESULTS_DIR / "streetview_location_comparison.csv"
    pd.DataFrame(location_rows).to_csv(location_csv, index=False, encoding="utf-8-sig")

    # Weight tuning and ranks
    tuning_df = image_df[[
        "filename", "baseline_walkability_score", "alternative_walkability_score"
    ]].copy()
    tuning_df = tuning_df.rename(columns={
        "baseline_walkability_score": "baseline_score",
        "alternative_walkability_score": "alternative_score",
    })
    tuning_df["baseline_weights"] = json.dumps(BASELINE_WEIGHTS)
    tuning_df["alternative_weights"] = json.dumps(ALTERNATIVE_WEIGHTS)
    tuning_df["baseline_rank"] = tuning_df["baseline_score"].rank(
        method="min", ascending=False
    ).astype(int)
    tuning_df["alternative_rank"] = tuning_df["alternative_score"].rank(
        method="min", ascending=False
    ).astype(int)
    tuning_df["rank_changed"] = tuning_df["baseline_rank"] != tuning_df["alternative_rank"]

    tuning_csv = RESULTS_DIR / "walkability_weight_comparison.csv"
    tuning_df.to_csv(tuning_csv, index=False, encoding="utf-8-sig")

    print("\nTop 5 under baseline weights:")
    print(tuning_df.sort_values("baseline_rank")[
        ["filename", "baseline_score", "baseline_rank"]
    ].head(5).to_string(index=False))

    print("\nTop 5 under alternative weights:")
    print(tuning_df.sort_values("alternative_rank")[
        ["filename", "alternative_score", "alternative_rank"]
    ].head(5).to_string(index=False))

    print(
        "\nImages whose rank changed:",
        int(tuning_df["rank_changed"].sum()), "out of", len(tuning_df)
    )
    print("\nOutputs:")
    print(image_csv)
    print(location_json)
    print(location_csv)
    print(tuning_csv)
    print(METADATA_CSV)
    print("\nNote: walkability scores are exploratory image-based indicators, not validated safety ratings.")


if __name__ == "__main__":
    main()
