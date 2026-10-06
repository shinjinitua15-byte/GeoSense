from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import xy
from shapely.geometry import Point


CLEAN_DIR = Path("data/satellite/clean")
RAW_DIR = Path("data/satellite/raw")

STUDY_AREA = Path(
    "data/shapefiles/AMC UPDATED/AMC_EDIT.shp"
)

CHIP_SIZE = 224
HALF = CHIP_SIZE // 2

# B2, B3, B4, B8, B11, B12
B4 = 2
B8 = 3


def compute_ndvi(data):
    nir = data[B8]
    red = data[B4]

    denominator = nir + red

    return np.divide(
        nir - red,
        denominator,
        out=np.zeros_like(
            nir,
            dtype=np.float32
        ),
        where=denominator != 0
    )


def get_patch(ndvi, row, col):
    return ndvi[
        row - HALF:row + HALF,
        col - HALF:col + HALF
    ]


# ---------------------------------------------------------
# Load NDVI data
# ---------------------------------------------------------

ndvi_2015 = compute_ndvi(
    np.load(
        CLEAN_DIR / "clean_2015.npy"
    )
)

ndvi_2023 = compute_ndvi(
    np.load(
        CLEAN_DIR / "clean_2023.npy"
    )
)


# ---------------------------------------------------------
# Raster information
# ---------------------------------------------------------

with rasterio.open(
    RAW_DIR / "sentinel2_AMC_2023.tif"
) as src:

    raster_transform = src.transform
    raster_crs = src.crs
    height = src.height
    width = src.width


# ---------------------------------------------------------
# AMC boundary
# ---------------------------------------------------------

gdf = gpd.read_file(
    STUDY_AREA
).to_crs(raster_crs)

geometry = gdf.geometry.iloc[0]


best_point = None
best_max_difference = float("inf")

STEP = 10

for row in range(
    HALF,
    height - HALF,
    STEP
):

    for col in range(
        HALF,
        width - HALF,
        STEP
    ):

        x, y = xy(
            raster_transform,
            row,
            col,
            offset="center"
        )

        if not geometry.contains(
            Point(x, y)
        ):
            continue

        patch15 = get_patch(
            ndvi_2015,
            row,
            col
        )

        patch23 = get_patch(
            ndvi_2023,
            row,
            col
        )

        difference = np.abs(
            patch23 - patch15
        )

        max_difference = float(
            np.max(difference)
        )

        if max_difference < best_max_difference:

            best_max_difference = max_difference

            best_point = (
                row,
                col,
                x,
                y,
                float(np.mean(patch15)),
                float(np.mean(patch23)),
                float(np.mean(patch23 - patch15))
            )


# ---------------------------------------------------------
# Print result
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("BEST UNCHANGED 224x224 PATCH")
print("=" * 65)

if best_point is None:

    print("No valid patch found.")

else:

    row, col, x, y, ndvi15, ndvi23, mean_diff = (
        best_point
    )

    point = gpd.GeoDataFrame(
        geometry=[Point(x, y)],
        crs=raster_crs
    ).to_crs("EPSG:4326")

    location = point.geometry.iloc[0]

    print(
        "Latitude:",
        f"{location.y:.6f}"
    )

    print(
        "Longitude:",
        f"{location.x:.6f}"
    )

    print(
        "NDVI 2015:",
        f"{ndvi15:.4f}"
    )

    print(
        "NDVI 2023:",
        f"{ndvi23:.4f}"
    )

    print(
        "Mean NDVI difference:",
        f"{mean_diff:.4f}"
    )

    print(
        "Maximum absolute pixel-level difference:",
        f"{best_max_difference:.4f}"
    )

    print(
        "Change flag should be:",
        "FALSE"
        if best_max_difference <= 0.15
        else "TRUE"
    )