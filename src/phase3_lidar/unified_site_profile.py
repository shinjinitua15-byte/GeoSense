
"""
unified_site_profile.py
GeoSense Agent 2.0 - Phase III

Tasks 66-69:
- EnhancedSiteProfile dataclass
- Unified LiDAR and street-view profile
- JSON report generation
- Explicit Phase 1 and Phase 2 integration fields

Important:
The available LiDAR data covers a site in the United States.
The existing street-view imagery covers Chennai, India.
They must not be treated as observations of the same location.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import sys

import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(r"D:\GeoSense_Phase3")
SCRIPT_DIR = PROJECT_DIR / "src" / "phase3_lidar"

RESULTS_DIR = PROJECT_DIR / "data" / "streetview" / "results"
METADATA_CSV = RESULTS_DIR / "mapillary_image_metadata.csv"
SEMANTIC_RESULTS_CSV = RESULTS_DIR / "streetview_semantic_results.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "reports"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lidar_analyzer import analyze_bbox
import streetview_semantic as streetview


# --------------------------------------------------
# Actual LiDAR study-area centre
# --------------------------------------------------
# Derived from the geographic bounds of lidar_hag.laz.
# These coordinates are NOT the Chennai street-view location.

DEFAULT_LATITUDE = 45.38575
DEFAULT_LONGITUDE = -114.03424
DEFAULT_RADIUS_DEG = 0.006


# --------------------------------------------------
# Enhanced site profile
# --------------------------------------------------

@dataclass
class EnhancedSiteProfile:
    """Structured profile containing spatial-analysis metrics."""

    latitude: float
    longitude: float
    avg_building_height_m: float | None
    canopy_cover_pct: float | None
    impervious_surface_pct: float | None
    slope_risk_score: float | None
    scene_composition: dict
    walkability_score: float | None


# --------------------------------------------------
# Load street-view results
# --------------------------------------------------

def load_streetview_data():
    """Load semantic class percentages and image coordinates."""

    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Street-view metadata not found: {METADATA_CSV}"
        )

    if not SEMANTIC_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Street-view semantic results not found: "
            f"{SEMANTIC_RESULTS_CSV}"
        )

    metadata = pd.read_csv(METADATA_CSV)
    results = pd.read_csv(SEMANTIC_RESULTS_CSV)

    required_metadata = {"filename", "latitude", "longitude"}
    missing = required_metadata - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing)}"
        )

    if "filename" not in results.columns:
        raise ValueError(
            "Semantic results must contain a filename column."
        )

    class_names = list(streetview.CLASS_NAMES.values())
    missing_classes = set(class_names) - set(results.columns)

    if missing_classes:
        raise ValueError(
            f"Missing semantic class columns: {sorted(missing_classes)}"
        )

    metadata["latitude"] = pd.to_numeric(
        metadata["latitude"], errors="coerce"
    )
    metadata["longitude"] = pd.to_numeric(
        metadata["longitude"], errors="coerce"
    )

    metadata = metadata.dropna(
        subset=["latitude", "longitude", "filename"]
    ).copy()

    compositions_by_filename = {}

    for _, row in results.iterrows():
        filename = str(row["filename"])
        composition = {}

        for class_name in class_names:
            value = pd.to_numeric(row[class_name], errors="coerce")

            if pd.isna(value):
                break

            composition[class_name] = float(value)
        else:
            compositions_by_filename[filename] = composition

    return compositions_by_filename, metadata


# --------------------------------------------------
# Build unified profile
# --------------------------------------------------

def build_profile(lat, lon, bbox_radius_deg=DEFAULT_RADIUS_DEG):
    """
    Combine LiDAR and nearby street-view metrics.

    Args:
        lat: Site-centre latitude in decimal degrees.
        lon: Site-centre longitude in decimal degrees.
        bbox_radius_deg: Half-width of the analysis bounding box.

    Returns:
        Dictionary containing the profile, data-quality notes,
        and Phase 1/Phase 2 integration status.

    Example:
        profile = build_profile(45.38575, -114.03424)
    """

    lat = float(lat)
    lon = float(lon)
    bbox_radius_deg = float(bbox_radius_deg)

    if not math.isfinite(lat) or not -90 <= lat <= 90:
        raise ValueError("Latitude must be finite and between -90 and 90.")

    if not math.isfinite(lon) or not -180 <= lon <= 180:
        raise ValueError(
            "Longitude must be finite and between -180 and 180."
        )

    if not math.isfinite(bbox_radius_deg) or bbox_radius_deg <= 0:
        raise ValueError("bbox_radius_deg must be a positive finite number.")

    # 1. Analyse the LiDAR point cloud.
    lidar = analyze_bbox(
        min_lat=lat - bbox_radius_deg,
        min_lon=lon - bbox_radius_deg,
        max_lat=lat + bbox_radius_deg,
        max_lon=lon + bbox_radius_deg,
    )

    lidar_count = int(lidar["total_points_analyzed"])
    lidar_available = lidar_count > 0

    # Missing measurements are None, not measured zero values.
    lidar_metrics = {
        "avg_building_height_m": (
            float(lidar["avg_building_height_m"])
            if lidar_available else None
        ),
        "canopy_cover_pct": (
            float(lidar["canopy_cover_pct"])
            if lidar_available else None
        ),
        "impervious_surface_pct": (
            float(lidar["impervious_surface_pct"])
            if lidar_available else None
        ),
        "slope_risk_score": (
            float(lidar["slope_risk_score"])
            if lidar_available else None
        ),
    }

    # 2. Load and analyse street-view data.
    compositions, metadata = load_streetview_data()

    street_result = streetview.analyze_location(
        lat,
        lon,
        compositions,
        metadata,
        radius_deg=bbox_radius_deg,
    )

    image_count = int(street_result.get("images_analyzed", 0) or 0)
    street_available = image_count > 0

    scene_composition = (
        street_result.get("scene_composition") or {}
    ) if street_available else {}

    walkability = (
        street_result.get("walkability_score")
        if street_available else None
    )

    # 3. Build the dataclass.
    profile = EnhancedSiteProfile(
        latitude=lat,
        longitude=lon,
        **lidar_metrics,
        scene_composition=scene_composition,
        walkability_score=(
            float(walkability) if walkability is not None else None
        ),
    )

    result = asdict(profile)

    # 4. Record source counts and bounding box.
    result["bbox_radius_deg"] = bbox_radius_deg
    result["lidar_points_analyzed"] = lidar_count
    result["streetview_images_analyzed"] = image_count
    result["streetview_images_used"] = (
        street_result.get("images_used", [])
        if street_available else []
    )
    result["lidar_bbox"] = lidar.get("bbox")

    # 5. Explicit Phase 1 and Phase 2 integration.
    # No Phase 1 suitability model/output or Phase 2
    # classify_imagery() implementation has been located yet.
    result["phase1_ml_suitability_score"] = None
    result["phase1_status"] = (
        "unavailable: model or prediction output not located"
    )

    result["phase2_imagery_classification"] = None
    result["phase2_status"] = (
        "unavailable: classify_imagery() implementation not located"
    )

    # Existing Phase 3 image segmentation is NOT the same as
    # the missing Phase 2 classify_imagery() function.
    result["phase3_streetview_semantic_analysis"] = {
        "available": street_available,
        "images_analyzed": image_count,
        "scene_composition": scene_composition,
        "walkability_score": result["walkability_score"],
    }

    # 6. Data-quality warnings.
    notes = []

    if not lidar_available:
        notes.append(
            "No LiDAR points were found in this bounding box. "
            "LiDAR metrics are null because no measurements are available."
        )

    if not street_available:
        notes.append(
            "No street-view images were found near the selected location. "
            "Walkability and scene composition are unavailable."
        )

    notes.append(
        "The available street-view metadata is from Chennai, India, "
        "whereas the LiDAR cloud is located in the United States. "
        "Only spatially matching observations should be combined."
    )

    notes.append(
        "Canopy and impervious percentages are point-share proxies from "
        "the current heuristic classification, not validated land-area "
        "cover estimates."
    )

    if result["phase1_ml_suitability_score"] is None:
        notes.append(
            "Phase 1 ML suitability score was not calculated because "
            "its model or actual prediction output was not found."
        )

    if result["phase2_imagery_classification"] is None:
        notes.append(
            "Phase 2 imagery classification is unavailable because "
            "classify_imagery() was not found in the inspected source files."
        )

    result["data_quality_notes"] = notes

    result["profile_status"] = (
        "incomplete"
        if (
            not lidar_available
            or not street_available
            or result["phase1_ml_suitability_score"] is None
            or result["phase2_imagery_classification"] is None
        )
        else "complete_pending_validation"
    )

    return result


# --------------------------------------------------
# Save JSON report
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    latitude = DEFAULT_LATITUDE
    longitude = DEFAULT_LONGITUDE

    print("=" * 64)
    print("GeoSense Agent 2.0 - Phase III Unified Site Profile")
    print("=" * 64)
    print(f"LiDAR site centre: {latitude}, {longitude}")
    print("Street-view results will only be included if coordinates match.")

    result = build_profile(latitude, longitude)

    output_file = (
        OUTPUT_DIR
        / f"enhanced_site_profile_{latitude:.5f}_{longitude:.5f}.json"
    )

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

    print("\nProfile status:", result["profile_status"])
    print("LiDAR points:", result["lidar_points_analyzed"])
    print("Street-view images:", result["streetview_images_analyzed"])
    print("Phase 1 suitability:", result["phase1_status"])
    print("Phase 2 imagery:", result["phase2_status"])
    print("\nJSON report:")
    print(json.dumps(result, indent=4, ensure_ascii=False))
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
