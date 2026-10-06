from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import xy
from shapely.geometry import Point


# ---------------------------------------------------------
# GeoSense Phase II
# Find changed/unchanged 224x224 patches
# ---------------------------------------------------------

CLEAN_DIR = Path("data/satellite/clean")
RAW_DIR = Path("data/satellite/raw")

STUDY_AREA = Path(
    "data/shapefiles/AMC UPDATED/AMC_EDIT.shp"
)

CHIP_SIZE = 224
HALF = CHIP_SIZE // 2

# Band positions:
# B2=0, B3=1, B4=2, B8=3, B11=4, B12=5
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


def patch_means(image, row, col):

    row_start = row - HALF
    row_end = row_start + CHIP_SIZE

    col_start = col - HALF
    col_end = col_start + CHIP_SIZE

    patch = image[
        row_start:row_end,
        col_start:col_end
    ]

    return float(np.nanmean(patch))


# ---------------------------------------------------------
# Load NDVI
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
# Open raster information
# ---------------------------------------------------------

raster_path = (
    RAW_DIR / "sentinel2_AMC_2023.tif"
)

with rasterio.open(raster_path) as src:

    transform = src.transform
    raster_crs = src.crs
    height = src.height
    width = src.width


# ---------------------------------------------------------
# Read AMC boundary
# ---------------------------------------------------------

gdf = gpd.read_file(
    STUDY_AREA
).to_crs(raster_crs)

if gdf.empty:
    raise ValueError(
        "AMC shapefile is empty."
    )

study_geometry = gdf.geometry.iloc[0]


# ---------------------------------------------------------
# Search possible patch centres
# ---------------------------------------------------------

best_changed = None
best_unchanged = None

# Search every 10 pixels.
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

        # Centre point of the 224x224 patch
        x, y = xy(
            transform,
            row,
            col,
            offset="center"
        )

        point = Point(
            x,
            y
        )

        if not study_geometry.contains(point):
            continue

        ndvi15 = patch_means(
            ndvi_2015,
            row,
            col
        )

        ndvi23 = patch_means(
            ndvi_2023,
            row,
            col
        )

        difference = ndvi23 - ndvi15
        absolute_difference = abs(
            difference
        )

        record = (
            row,
            col,
            x,
            y,
            ndvi15,
            ndvi23,
            difference,
            absolute_difference
        )

        # Largest NDVI difference
        if best_changed is None:
            best_changed = record

        elif absolute_difference > best_changed[7]:
            best_changed = record

        # Smallest NDVI difference
        if best_unchanged is None:
            best_unchanged = record

        elif absolute_difference < best_unchanged[7]:
            best_unchanged = record


# ---------------------------------------------------------
# Convert raster coordinates to WGS84
# ---------------------------------------------------------

def print_record(title, record):

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    if record is None:

        print(
            "No valid patch found."
        )

        return

    row, col, x, y, ndvi15, ndvi23, diff, absdiff = (
        record
    )

    point_gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y)],
        crs=raster_crs
    ).to_crs("EPSG:4326")

    point_wgs84 = (
        point_gdf.geometry.iloc[0]
    )

    lon = point_wgs84.x
    lat = point_wgs84.y

    print(
        "Latitude:",
        f"{lat:.6f}"
    )

    print(
        "Longitude:",
        f"{lon:.6f}"
    )

    print(
        "Patch size:",
        f"{CHIP_SIZE} x {CHIP_SIZE}"
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
        "NDVI difference:",
        f"{diff:.4f}"
    )

    print(
        "Absolute NDVI difference:",
        f"{absdiff:.4f}"
    )

    if absdiff > 0.15:

        print(
            "Change flag should be: TRUE"
        )

    else:

        print(
            "Change flag should be: FALSE"
        )


print_record(
    "LARGEST 224x224 NDVI CHANGE",
    best_changed
)

print_record(
    "SMALLEST 224x224 NDVI CHANGE",
    best_unchanged
)