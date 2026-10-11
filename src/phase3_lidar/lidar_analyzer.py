
"""
lidar_analyzer.py
GeoSense Agent 2.0 - Phase III
LiDAR analysis and unified site metrics.

Tasks covered:
- Task 60: Slope risk calculation
- Task 61: Bounding-box LiDAR analysis
- Task 62: Function documentation
- Task 63: Example execution
- Task 64: Geographic bbox filtering using the point-cloud CRS
"""

from pathlib import Path

import laspy
import numpy as np
from pyproj import Transformer


# --------------------------------------------------
# Project paths and classification thresholds
# --------------------------------------------------

PROJECT_DIR = Path(r"D:\GeoSense_Phase3")
PROCESSED_CLOUD_PATH = PROJECT_DIR / "outputs" / "lidar_hag.laz"

GROUND_MAX_HEIGHT = 0.5
VEGETATION_MAX_HEIGHT = 2.5


# --------------------------------------------------
# Point classification
# --------------------------------------------------

def classify_points(hag, number_of_returns, return_number):
    """
    Classify LiDAR points using height above ground and return information.

    Args:
        hag: Array of HeightAboveGround values in metres.
        number_of_returns: Total returns for each emitted pulse.
        return_number: Return number of each point.

    Returns:
        NumPy array of labels:
        0 = Ground, 1 = Vegetation, 2 = Building.
    """
    hag = np.asarray(hag)
    number_of_returns = np.asarray(number_of_returns)
    return_number = np.asarray(return_number)

    labels = np.zeros(len(hag), dtype=np.uint8)

    vegetation = (
        (hag >= GROUND_MAX_HEIGHT)
        & (
            (hag < VEGETATION_MAX_HEIGHT)
            | (number_of_returns > 1)
        )
    )

    building = (
        (hag >= VEGETATION_MAX_HEIGHT)
        & (number_of_returns <= 1)
    )

    labels[vegetation] = 1
    labels[building] = 2

    return labels


# --------------------------------------------------
# Slope risk
# --------------------------------------------------

def _compute_slope_risk(z_values):
    """
    Estimate relative slope risk from the variation in elevation.

    Args:
        z_values: Sequence or array of elevation values in metres.

    Returns:
        A score from 0 to 10. Returns 0.0 when fewer than
        two elevation values are available.

    Example:
        >>> _compute_slope_risk([100, 101, 102])
        1.632993161855452
    """
    z_values = np.asarray(z_values, dtype=float)
    z_values = z_values[np.isfinite(z_values)]

    if len(z_values) < 2:
        return 0.0

    score = float(np.std(z_values) * 2.0)
    return round(min(score, 10.0), 4)


# --------------------------------------------------
# Bounding-box analysis
# --------------------------------------------------

def analyze_bbox(min_lat, min_lon, max_lat, max_lon):
    """
    Analyze LiDAR points inside a geographic bounding box.

    The input coordinates are latitude/longitude in degrees.
    The box is transformed into the point cloud's CRS before
    points are selected.

    Args:
        min_lat: Minimum latitude in decimal degrees.
        min_lon: Minimum longitude in decimal degrees.
        max_lat: Maximum latitude in decimal degrees.
        max_lon: Maximum longitude in decimal degrees.

    Returns:
        Dictionary containing the bounding box, average
        building height, vegetation point-share proxy,
        building point-share proxy, slope-risk score and
        total points analyzed.

    Example:
        >>> result = analyze_bbox(45.38, -114.04, 45.39, -114.03)
        >>> print(result["total_points_analyzed"])
    """
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees.")

    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise ValueError("Longitude must be between -180 and 180 degrees.")

    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError(
            "Minimum latitude/longitude must be less than maximum values."
        )

    if not PROCESSED_CLOUD_PATH.exists():
        raise FileNotFoundError(
            f"LiDAR file not found: {PROCESSED_CLOUD_PATH}"
        )

    cloud = laspy.read(PROCESSED_CLOUD_PATH)

    source_crs = cloud.header.parse_crs()
    if source_crs is None:
        raise ValueError(
            "The LiDAR file has no CRS metadata. Assign the correct CRS "
            "before performing geographic bounding-box analysis."
        )

    # Convert the geographic bbox corners to the LiDAR CRS.
    transformer = Transformer.from_crs(
        "EPSG:4326",
        source_crs,
        always_xy=True
    )

    corners_lon = [min_lon, min_lon, max_lon, max_lon]
    corners_lat = [min_lat, max_lat, min_lat, max_lat]

    projected_x, projected_y = transformer.transform(
        corners_lon, corners_lat
    )

    min_x = min(projected_x)
    max_x = max(projected_x)
    min_y = min(projected_y)
    max_y = max(projected_y)

    # Select points within the projected rectangle.
    x = np.asarray(cloud.x)
    y = np.asarray(cloud.y)
    z = np.asarray(cloud.z)

    mask = (
        (x >= min_x)
        & (x <= max_x)
        & (y >= min_y)
        & (y <= max_y)
    )

    if not np.any(mask):
        return {
            "bbox": {
                "min_lat": min_lat,
                "min_lon": min_lon,
                "max_lat": max_lat,
                "max_lon": max_lon,
            },
            "avg_building_height_m": 0.0,
            "canopy_cover_pct": 0.0,
            "impervious_surface_pct": 0.0,
            "slope_risk_score": 0.0,
            "total_points_analyzed": 0,
        }

    hag = np.asarray(cloud["HeightAboveGround"], dtype=float)[mask]
    number_of_returns = np.asarray(
        cloud["number_of_returns"]
    )[mask]
    return_number = np.asarray(cloud["return_number"])[mask]
    selected_z = z[mask]

    labels = classify_points(hag, number_of_returns, return_number)

    ground_count = int(np.sum(labels == 0))
    vegetation_count = int(np.sum(labels == 1))
    building_count = int(np.sum(labels == 2))
    total_count = int(len(labels))

    building_heights = hag[labels == 2]
    avg_building_height = (
        float(np.mean(building_heights))
        if len(building_heights) > 0
        else 0.0
    )

    # These are point-share proxies, not direct measurements of land area.
    canopy_cover_pct = vegetation_count / total_count * 100.0
    impervious_surface_pct = building_count / total_count * 100.0

    return {
        "bbox": {
            "min_lat": min_lat,
            "min_lon": min_lon,
            "max_lat": max_lat,
            "max_lon": max_lon,
        },
        "avg_building_height_m": round(avg_building_height, 2),
        "canopy_cover_pct": round(canopy_cover_pct, 2),
        "impervious_surface_pct": round(impervious_surface_pct, 2),
        "slope_risk_score": _compute_slope_risk(selected_z),
        "total_points_analyzed": total_count,
    }


# --------------------------------------------------
# Example execution
# --------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("GeoSense Phase III - LiDAR Analyzer")
    print("=" * 60)

    # The practicum's Chennai example.
    # The current LiDAR dataset is near longitude -114, latitude 45,
    # so this Chennai box is expected to contain zero points.
    chennai_bbox = (13.05, 80.24, 13.06, 80.25)

    # A second box near the actual LiDAR coverage.
    local_bbox = (45.38, -114.04, 45.39, -114.03)

    for name, bbox in [
        ("Practicum Chennai example", chennai_bbox),
        ("Actual LiDAR coverage", local_bbox),
    ]:
        print(f"\n{name}")
        print("-" * 40)

        result = analyze_bbox(*bbox)

        for key, value in result.items():
            print(f"{key}: {value}")
