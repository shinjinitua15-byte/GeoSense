# feature_engineering.py
# Purpose: Calculate geospatial features for every candidate location.

import geopandas as gpd
import pandas as pd
import numpy as np

from db_connection import get_engine


# WGS 84 / UTM Zone 46N - metric CRS for Agartala
# Distance calculations are performed in metres.
METRIC_CRS = 32646


def load_layer(sql, engine):
    """Load a spatial layer from PostGIS."""
    return gpd.read_postgis(
        sql,
        engine,
        geom_col="geometry"
    )


def calc_distance_to_nearest(candidates_gdf, layer_gdf):
    """Calculate distance in metres to the nearest feature."""

    candidates_projected = candidates_gdf.to_crs(
        epsg=METRIC_CRS
    )

    layer_projected = layer_gdf.to_crs(
        epsg=METRIC_CRS
    )

    distances = candidates_projected.geometry.apply(
        lambda point: layer_projected.geometry.distance(point).min()
    )

    return distances


def load_schools(engine):
    """
    Load schools from both OSM POI tables.
    Point schools are used directly.
    Area schools are converted to centroids in the metric CRS.
    """

    # School point features
    points = load_layer(
        """
        SELECT geometry
        FROM osm_poi
        WHERE LOWER(fclass) = 'school'
        """,
        engine
    )

    # School area features
    areas = load_layer(
        """
        SELECT geometry
        FROM osm_poi_area
        WHERE LOWER(fclass) = 'school'
        """,
        engine
    )

    # Project to metric CRS before calculating centroids
    areas = areas.to_crs(epsg=METRIC_CRS)

    # Convert area schools to centroids
    areas["geometry"] = areas.geometry.centroid

    # Return to WGS 84
    areas = areas.to_crs(epsg=4326)

    # Combine both school representations
    schools = pd.concat(
        [points, areas],
        ignore_index=True
    )

    print(
        f"Schools assembled: "
        f"{len(points)} point schools + "
        f"{len(areas)} area schools = "
        f"{len(schools)} total schools"
    )

    return gpd.GeoDataFrame(
        schools,
        geometry="geometry",
        crs="EPSG:4326"
    )


def assign_flood_risk(candidates_gdf, engine):
    """
    Assign flood-risk scores.
    High = 8, Medium = 5, Low = 2.
    If flood_zones is unavailable, use 0 as a placeholder.
    """

    try:

        flood = load_layer(
            """
            SELECT geometry, risk_level
            FROM flood_zones
            """,
            engine
        )

        joined = gpd.sjoin(
            candidates_gdf,
            flood,
            how="left",
            predicate="within"
        )

        risk_map = {
            "high": 8,
            "medium": 5,
            "low": 2
        }

        return joined["risk_level"].map(
            risk_map
        ).fillna(0)

    except Exception:

        print(
            "flood_zones table not found - "
            "flood_risk set to 0 (placeholder)"
        )

        return pd.Series(
            np.zeros(len(candidates_gdf))
        )


def build_feature_table(candidates_gdf, engine):

    # Basic candidate attributes
    df = candidates_gdf[
        ["location_id", "latitude", "longitude"]
    ].copy()

    # Distance to nearest road
    print("Calculating distance to nearest road...")

    roads = load_layer(
        """
        SELECT geometry
        FROM osm_roads
        """,
        engine
    )

    df["dist_road_m"] = calc_distance_to_nearest(
        candidates_gdf,
        roads
    )

    # Distance to nearest school
    print("Calculating distance to nearest school...")

    schools = load_schools(engine)

    df["dist_school_m"] = calc_distance_to_nearest(
        candidates_gdf,
        schools
    )

    # Flood-risk score
    print("Calculating flood risk score...")

    df["flood_risk"] = assign_flood_risk(
        candidates_gdf,
        engine
    )

    print(
        "Feature table built with",
        len(df),
        "locations and",
        len(df.columns),
        "columns"
    )

    return df


if __name__ == "__main__":

    engine = get_engine()

    # Read candidate locations
    candidates = gpd.read_file(
        "data/processed/candidate_locations.shp"
    )

    # Restore the field name shortened by Shapefile format
    candidates = candidates.rename(
        columns={
            "location_i": "location_id"
        }
    )

    # Build feature table
    features = build_feature_table(
        candidates,
        engine
    )

    # Save feature table
    features.to_csv(
        "data/processed/features.csv",
        index=False
    )

    print(
        "Features saved to data/processed/features.csv"
    )