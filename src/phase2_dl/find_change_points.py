from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import xy


CLEAN_DIR = Path("data/satellite/clean")
RAW_DIR = Path("data/satellite/raw")

STUDY_AREA = Path(
    "data/shapefiles/AMC UPDATED/AMC_EDIT.shp"
)

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


# ---------------------------------------------------------
# Load data
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

difference = ndvi_2023 - ndvi_2015
absolute_difference = np.abs(difference)


# ---------------------------------------------------------
# Get raster transform
# ---------------------------------------------------------

raster_path = (
    RAW_DIR / "sentinel2_AMC_2023.tif"
)

with rasterio.open(raster_path) as src:
    raster_transform = src.transform
    raster_crs = src.crs


# ---------------------------------------------------------
# Find pixels inside AMC
# ---------------------------------------------------------

gdf = gpd.read_file(STUDY_AREA)

if gdf.empty:
    raise ValueError("AMC shapefile is empty.")

gdf = gdf.to_crs(raster_crs)

geometry = gdf.geometry.iloc[0]


changed_candidates = []
unchanged_candidates = []


height, width = difference.shape


# Sample every 5th pixel to make the search faster
for row in range(0, height, 5):

    for col in range(0, width, 5):

        diff = absolute_difference[
            row,
            col
        ]

        if not np.isfinite(diff):
            continue

        x, y = xy(
            raster_transform,
            row,
            col,
            offset="center"
        )

        point = gpd.GeoSeries(
            [gpd.points_from_xy([x], [y])[0]],
            crs=raster_crs
        ).iloc[0]

        if not geometry.contains(point):
            continue

        record = (
            row,
            col,
            x,
            y,
            float(ndvi_2015[row, col]),
            float(ndvi_2023[row, col]),
            float(diff)
        )

        if diff > 0.15:
            changed_candidates.append(record)

        elif diff < 0.05:
            unchanged_candidates.append(record)


# ---------------------------------------------------------
# Print changed location
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CHANGED LOCATION")
print("=" * 60)

if changed_candidates:

    changed_candidates.sort(
        key=lambda x: x[6],
        reverse=True
    )

    row, col, x, y, ndvi15, ndvi23, diff = (
        changed_candidates[0]
    )

    # Convert raster CRS coordinates to WGS84
    point_gdf = gpd.GeoDataFrame(
        geometry=[
            gpd.points_from_xy([x], [y])[0]
        ],
        crs=raster_crs
    ).to_crs("EPSG:4326")

    lon = point_gdf.geometry.iloc[0].x
    lat = point_gdf.geometry.iloc[0].y

    print("Latitude:", round(lat, 6))
    print("Longitude:", round(lon, 6))
    print("NDVI 2015:", round(ndvi15, 4))
    print("NDVI 2023:", round(ndvi23, 4))
    print("Absolute NDVI difference:", round(diff, 4))
    print("Change flag should be: TRUE")

else:
    print(
        "No changed pixel found with "
        "absolute NDVI difference > 0.15."
    )


# ---------------------------------------------------------
# Print unchanged location
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("UNCHANGED LOCATION")
print("=" * 60)

if unchanged_candidates:

    unchanged_candidates.sort(
        key=lambda x: x[6]
    )

    row, col, x, y, ndvi15, ndvi23, diff = (
        unchanged_candidates[0]
    )

    point_gdf = gpd.GeoDataFrame(
        geometry=[
            gpd.points_from_xy([x], [y])[0]
        ],
        crs=raster_crs
    ).to_crs("EPSG:4326")

    lon = point_gdf.geometry.iloc[0].x
    lat = point_gdf.geometry.iloc[0].y

    print("Latitude:", round(lat, 6))
    print("Longitude:", round(lon, 6))
    print("NDVI 2015:", round(ndvi15, 4))
    print("NDVI 2023:", round(ndvi23, 4))
    print("Absolute NDVI difference:", round(diff, 4))
    print("Change flag should be: FALSE")

else:
    print(
        "No unchanged pixel found."
    )