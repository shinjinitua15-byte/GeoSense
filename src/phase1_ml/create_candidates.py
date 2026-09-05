import os
import geopandas as gpd
import numpy as np
from shapely.geometry import Point


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
AMC_PATH = r"data/shapefiles/AMC UPDATED/AMC_EDIT.shp"
OUTPUT_PATH = r"data/processed/candidate_locations.shp"


# ---------------------------------------------------------
# Candidate grid settings
# ---------------------------------------------------------
GRID_SPACING = 0.005   # approximately 500 m


def main():

    # Create output folder if it does not exist
    os.makedirs("data/processed", exist_ok=True)

    # -----------------------------------------------------
    # 1. Read AMC boundary
    # -----------------------------------------------------
    print("Reading AMC boundary...")

    amc = gpd.read_file(AMC_PATH)

    print("AMC CRS:", amc.crs)
    print("AMC features:", len(amc))

    # Convert to geographic CRS
    amc_wgs84 = amc.to_crs(epsg=4326)

    # Merge boundary features
    amc_geometry = amc_wgs84.union_all()

    # -----------------------------------------------------
    # 2. Get AMC bounding box
    # -----------------------------------------------------
    minx, miny, maxx, maxy = amc_geometry.bounds

    print("AMC bounds:")
    print("Minimum longitude:", minx)
    print("Minimum latitude :", miny)
    print("Maximum longitude:", maxx)
    print("Maximum latitude :", maxy)

    # -----------------------------------------------------
    # 3. Generate regular grid points
    # -----------------------------------------------------
    print("Generating candidate points...")

    longitudes = np.arange(minx, maxx + GRID_SPACING, GRID_SPACING)
    latitudes = np.arange(miny, maxy + GRID_SPACING, GRID_SPACING)

    candidate_points = []

    for lon in longitudes:
        for lat in latitudes:
            point = Point(lon, lat)

            # Keep only points inside AMC
            if amc_geometry.covers(point):
                candidate_points.append(point)

    # -----------------------------------------------------
    # 4. Create GeoDataFrame
    # -----------------------------------------------------
    candidates = gpd.GeoDataFrame(
        {
            "location_id": range(1, len(candidate_points) + 1),
            "longitude": [point.x for point in candidate_points],
            "latitude": [point.y for point in candidate_points],
        },
        geometry=candidate_points,
        crs="EPSG:4326"
    )

    # -----------------------------------------------------
    # 5. Save candidate points
    # -----------------------------------------------------
    candidates.to_file(
        OUTPUT_PATH,
        driver="ESRI Shapefile"
    )

    # -----------------------------------------------------
    # 6. Display summary
    # -----------------------------------------------------
    print("\nCandidate grid generated successfully!")
    print("Total candidate locations:", len(candidates))
    print("Output file:", OUTPUT_PATH)


if __name__ == "__main__":
    main()