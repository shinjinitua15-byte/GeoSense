
from pathlib import Path

import laspy
import numpy as np
import pandas as pd


# ==========================================================
# GeoSense Agent 2.0 - Phase III
# Exercise 3.1: LiDAR Feature Extraction and Grid Metrics
# ==========================================================

PROJECT_DIR = Path(r"D:\GeoSense_Phase3")
INPUT_FILE = PROJECT_DIR / "outputs" / "lidar_hag.laz"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "lidar_features"

GRID_SIZE_M = 50.0


def classify_points(hag, num_returns, return_num):
    """
    Classify LiDAR points according to the practicum rules.

    Labels:
        0 = Ground
        1 = Vegetation
        2 = Building

    Rules:
        HAG < 0.5 m              -> Ground
        0.5 <= HAG < 2.5 m       -> Vegetation
        HAG >= 2.5 m and multiple
        returns                  -> Vegetation
        Remaining single-return
        points                   -> Building
    """
    hag = np.asarray(hag)
    num_returns = np.asarray(num_returns)
    return_num = np.asarray(return_num)

    if not (hag.shape == num_returns.shape == return_num.shape):
        raise ValueError("HAG and return arrays must have matching shapes.")

    # Default class: building
    labels = np.full(hag.shape, 2, dtype=np.uint8)

    # Ground
    labels[hag < 0.5] = 0

    # Low vegetation
    low_vegetation = (hag >= 0.5) & (hag < 2.5)
    labels[low_vegetation] = 1

    # Tall vegetation with multiple returns
    tall_vegetation = (hag >= 2.5) & (num_returns > 1)
    labels[tall_vegetation] = 1

    return labels


def compute_grid_metrics(
    x,
    y,
    hag,
    labels,
    cell_size_m=50.0
):
    """
    Calculate metrics for each 50 m grid cell.

    avg_building_height_m:
        Mean HAG of points labelled as buildings.

    canopy_cover_pct:
        Percentage of points labelled as vegetation.
        This is point share, not a direct area-based
        canopy-cover measurement.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    hag = np.asarray(hag)
    labels = np.asarray(labels)

    if len(x) == 0:
        raise ValueError("No valid points are available for grid analysis.")

    if cell_size_m <= 0:
        raise ValueError("Cell size must be greater than zero.")

    # Align the grid origin to multiples of cell_size_m
    min_x = np.floor(x.min() / cell_size_m) * cell_size_m
    min_y = np.floor(y.min() / cell_size_m) * cell_size_m

    grid_x = np.floor((x - min_x) / cell_size_m).astype(np.int32)
    grid_y = np.floor((y - min_y) / cell_size_m).astype(np.int32)

    # Group points by grid cell
    point_table = pd.DataFrame({
        "grid_x": grid_x,
        "grid_y": grid_y,
        "hag": hag,
        "label": labels
    })

    grouped = point_table.groupby(
        ["grid_x", "grid_y"],
        sort=True
    )

    records = []

    for (gx, gy), group in grouped:
        building_heights = group.loc[
            group["label"] == 2, "hag"
        ]

        vegetation_count = int(
            (group["label"] == 1).sum()
        )

        total_points = len(group)

        canopy_cover_pct = (
            vegetation_count / total_points * 100.0
        )

        average_building_height = (
            float(building_heights.mean())
            if len(building_heights) > 0
            else np.nan
        )

        records.append({
            "grid_x": int(gx),
            "grid_y": int(gy),
            "avg_building_height_m": average_building_height,
            "canopy_cover_pct": float(canopy_cover_pct),
            "point_count": int(total_points)
        })

    return pd.DataFrame(records)


def main():
    # Check input file
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GEOSENSE PHASE III - EXERCISE 3.1")
    print("LiDAR Feature Extraction and Grid Metrics")
    print("=" * 60)

    # Read the point cloud
    print("\nReading LiDAR point cloud...")

    with laspy.open(str(INPUT_FILE)) as reader:
        cloud = reader.read()

    # IMPORTANT:
    # laspy exposes LAS dimension names in lowercase,
    # except for custom dimensions such as HeightAboveGround.
    available_dimensions = set(
        cloud.point_format.dimension_names
    )

    required_dimensions = [
        "X",
        "Y",
        "Z",
        "HeightAboveGround",
        "number_of_returns",
        "return_number"
    ]

    missing_dimensions = [
        name for name in required_dimensions
        if name not in available_dimensions
    ]

    if missing_dimensions:
        raise ValueError(
            "Required dimensions are missing: "
            f"{missing_dimensions}\n"
            f"Available dimensions: {sorted(available_dimensions)}"
        )

    # Extract coordinate and attribute arrays
    x = np.asarray(cloud.x)
    y = np.asarray(cloud.y)
    hag = np.asarray(cloud["HeightAboveGround"])
    num_returns = np.asarray(cloud.number_of_returns)
    return_num = np.asarray(cloud.return_number)

    # Exclude points with invalid coordinate/HAG values
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(hag)
    )

    x = x[valid]
    y = y[valid]
    hag = hag[valid]
    num_returns = num_returns[valid]
    return_num = return_num[valid]

    if len(x) == 0:
        raise ValueError("No valid LiDAR points remain after filtering.")

    print(f"Valid points processed: {len(x):,}")

    # Classify points
    print("\nClassifying points...")

    labels = classify_points(
        hag,
        num_returns,
        return_num
    )

    class_names = {
        0: "Ground",
        1: "Vegetation",
        2: "Building"
    }

    total_points = len(labels)

    for code, name in class_names.items():
        count = int(np.count_nonzero(labels == code))
        percentage = count / total_points * 100.0

        print(
            f"{name}: {count:,} points "
            f"({percentage:.2f}%)"
        )

    # Save classification summary
    summary_records = []

    for code, name in class_names.items():
        count = int(np.count_nonzero(labels == code))

        summary_records.append({
            "class_code": code,
            "class_name": name,
            "point_count": count,
            "percentage": count / total_points * 100.0
        })

    summary_df = pd.DataFrame(summary_records)

    summary_path = (
        OUTPUT_DIR / "point_classification_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    # Calculate 50 m grid metrics
    print(f"\nCalculating {GRID_SIZE_M:.0f} m grid metrics...")

    grid_metrics = compute_grid_metrics(
        x=x,
        y=y,
        hag=hag,
        labels=labels,
        cell_size_m=GRID_SIZE_M
    )

    grid_path = OUTPUT_DIR / "lidar_grid_metrics_50m.csv"
    grid_metrics.to_csv(grid_path, index=False)

    # Display required results
    print(f"Grid cell count: {len(grid_metrics):,}")

    print("\nFirst three grid-cell results:")
    print(grid_metrics.head(3).to_string(index=False))

    print("\nOutput files created:")
    print(summary_path)
    print(grid_path)

    print("\nInterpretation notes:")
    print(
        "1. Classification is a height-and-return heuristic, "
        "not a guaranteed semantic classification."
    )
    print(
        "2. Low objects can be labelled ground, while tall "
        "single-return trees can be labelled buildings."
    )
    print(
        "3. Canopy cover represents vegetation point share, "
        "not measured canopy area."
    )
    print(
        "4. Grid X and Y values are grid indices, not "
        "geographic coordinates."
    )

    print("\nExercise 3.1 processing finished.")


if __name__ == "__main__":
    main()
