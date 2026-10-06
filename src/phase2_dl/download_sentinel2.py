import os
from pathlib import Path

import ee
import geemap
import geopandas as gpd
import rasterio
from rasterio.merge import merge


# ---------------------------------------------------------
# GeoSense Phase II - Exercise 2.1
# Sentinel-2 data acquisition
# ---------------------------------------------------------

PROJECT_ID = "summer-ranger-448906-k0"

STUDY_AREA_PATH = r"data\shapefiles\AMC UPDATED\AMC_EDIT.shp"

OUTPUT_DIR = Path("data/satellite/raw")
TILE_DIR = OUTPUT_DIR / "tiles"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TILE_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2015, 2023]


def initialise_earth_engine():
    ee.Initialize(project=PROJECT_ID)
    print("Google Earth Engine initialized successfully.")


def get_study_area():
    gdf = gpd.read_file(STUDY_AREA_PATH)

    if gdf.empty:
        raise ValueError("Study-area shapefile contains no features.")

    print("Study area CRS:", gdf.crs)
    print("Study area features:", len(gdf))

    gdf = gdf.to_crs("EPSG:4326")

    return geemap.geopandas_to_ee(gdf)


def get_clean_image(year, study_area):

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(study_area.geometry())
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    )

    count = collection.size().getInfo()

    print(f"{year} matching images: {count}")

    if count == 0:
        raise ValueError(f"No Sentinel-2 images found for {year}.")

    image = collection.median()

    image = image.select(
        ["B2", "B3", "B4", "B8", "B11", "B12"]
    )

    return image.clip(study_area.geometry())


def create_export_grid(study_area):
    """Create a 2x2 grid covering the study area."""

    bounds = study_area.geometry().bounds().coordinates().getInfo()[0]

    xs = [p[0] for p in bounds]
    ys = [p[1] for p in bounds]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2

    cells = [
        ee.Geometry.Rectangle([xmin, ymin, xmid, ymid]),
        ee.Geometry.Rectangle([xmid, ymin, xmax, ymid]),
        ee.Geometry.Rectangle([xmin, ymid, xmid, ymax]),
        ee.Geometry.Rectangle([xmid, ymid, xmax, ymax]),
    ]

    return cells


def merge_tiles(year, tile_files, final_file):

    src_files = []

    try:
        for file in tile_files:
            src = rasterio.open(file)
            src_files.append(src)

        mosaic, transform = merge(src_files)

        profile = src_files[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform
        )

        with rasterio.open(final_file, "w", **profile) as dst:
            dst.write(mosaic)

    finally:
        for src in src_files:
            src.close()

    print(f"Merged output created: {final_file}")


def download_composite(year, study_area):

    image = get_clean_image(year, study_area)

    year_tile_dir = TILE_DIR / str(year)
    year_tile_dir.mkdir(parents=True, exist_ok=True)

    grid = create_export_grid(study_area)

    tile_files = []

    print(f"\nDownloading {year} composite in 4 tiles...")

    for i, cell in enumerate(grid, start=1):

        tile_file = year_tile_dir / f"tile_{i}.tif"

        print(f"Downloading tile {i}/4...")

        geemap.ee_export_image(
            image,
            filename=str(tile_file),
            scale=10,
            region=cell,
            file_per_band=False
        )

        if tile_file.exists() and tile_file.stat().st_size > 0:
            tile_files.append(tile_file)
            print(f"Tile {i} downloaded successfully.")
        else:
            raise RuntimeError(f"Tile {i} was not downloaded correctly.")

    final_file = OUTPUT_DIR / f"sentinel2_AMC_{year}.tif"

    merge_tiles(year, tile_files, final_file)

    print(f"{year} composite completed.\n")


def main():

    initialise_earth_engine()

    print("\nReading AMC study area...")
    study_area = get_study_area()

    print("\nStarting Sentinel-2 downloads...\n")

    for year in YEARS:
        download_composite(year, study_area)

    print("All Sentinel-2 composites created successfully.")
    print(f"Final files are stored in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()