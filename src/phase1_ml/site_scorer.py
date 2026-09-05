# site_scorer.py
# Purpose: Package the trained model into a reusable scoring function.

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import shap

from shapely.geometry import Point
from db_connection import get_engine


# WGS 84 / UTM Zone 46N - metric CRS for Agartala
METRIC_CRS = 32646

# Same features used during model training
FEATURE_NAMES = [
    "dist_road_m",
    "dist_school_m",
    "flood_risk"
]


# ---------------------------------------------------------
# Load the trained model only when first required
# ---------------------------------------------------------

_model = None
_explainer = None


def _load_model():
    global _model
    global _explainer

    if _model is None:
        _model = joblib.load(
            "models/saved/site_scorer_model.pkl"
        )

        _explainer = shap.TreeExplainer(
            _model
        )

    return _model, _explainer


# ---------------------------------------------------------
# Calculate features for a single location
# ---------------------------------------------------------

def _get_features_for_point(lat, lon, engine):

    """Calculate model features for one latitude/longitude."""

    point_gdf = gpd.GeoDataFrame(
        [
            {
                "geometry": Point(lon, lat)
            }
        ],
        crs="EPSG:4326"
    )

    # Project candidate point to metric CRS
    point_projected = point_gdf.to_crs(
        epsg=METRIC_CRS
    ).geometry.iloc[0]

    # -----------------------------------------------------
    # Distance to nearest road
    # -----------------------------------------------------

    roads = gpd.read_postgis(
        "SELECT geometry FROM osm_roads",
        engine,
        geom_col="geometry"
    ).to_crs(
        epsg=METRIC_CRS
    )

    dist_road = float(
        roads.geometry.distance(
            point_projected
        ).min()
    )

    # -----------------------------------------------------
    # School point features
    # -----------------------------------------------------

    school_points = gpd.read_postgis(
        """
        SELECT geometry
        FROM osm_poi
        WHERE LOWER(fclass) = 'school'
        """,
        engine,
        geom_col="geometry"
    ).to_crs(
        epsg=METRIC_CRS
    )

    # -----------------------------------------------------
    # School area features
    # -----------------------------------------------------

    school_areas = gpd.read_postgis(
        """
        SELECT geometry
        FROM osm_poi_area
        WHERE LOWER(fclass) = 'school'
        """,
        engine,
        geom_col="geometry"
    ).to_crs(
        epsg=METRIC_CRS
    )

    # Convert school polygons to centroids
    school_areas["geometry"] = (
        school_areas.geometry.centroid
    )

    # Combine point and area schools
    schools = pd.concat(
        [school_points, school_areas],
        ignore_index=True
    )

    # -----------------------------------------------------
    # Distance to nearest school
    # -----------------------------------------------------

    dist_school = float(
        schools.geometry.distance(
            point_projected
        ).min()
    )

    # -----------------------------------------------------
    # Flood risk
    # -----------------------------------------------------

    # Placeholder, same as training dataset
    flood_risk = 0.0

    return {
        "dist_road_m": dist_road,
        "dist_school_m": dist_school,
        "flood_risk": flood_risk
    }


# ---------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------

def score_location(lat: float, lon: float) -> dict:

    """
    Score a location using the trained model.

    Returns:
        latitude
        longitude
        opportunity_score
        risk_score
        features
        shap_explanation
        verdict
    """

    engine = get_engine()

    model, explainer = _load_model()

    # Calculate features
    features = _get_features_for_point(
        lat,
        lon,
        engine
    )

    # Use DataFrame with the exact training feature names
    X = pd.DataFrame(
        [
            [
                features["dist_road_m"],
                features["dist_school_m"],
                features["flood_risk"]
            ]
        ],
        columns=FEATURE_NAMES
    )

    # Model probability
    probabilities = model.predict_proba(X)[0]

    # Convert probability into 0-10 scores
    opportunity_score = round(
        float(probabilities[1]) * 10,
        2
    )

    risk_score = round(
        float(probabilities[0]) * 10,
        2
    )

    # -----------------------------------------------------
    # SHAP explanation
    # -----------------------------------------------------

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1][0]

    elif getattr(
        shap_values,
        "ndim",
        2
    ) == 3:
        shap_values = shap_values[0, :, 1]

    else:
        shap_values = shap_values[0]

    explanation = {
        name: round(
            float(value),
            4
        )
        for name, value in zip(
            FEATURE_NAMES,
            shap_values
        )
    }

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    verdict = (
        "GOOD SITE"
        if opportunity_score > 5
        else "POOR SITE"
    )

    return {
        "latitude": lat,
        "longitude": lon,
        "opportunity_score": opportunity_score,
        "risk_score": risk_score,
        "features": features,
        "shap_explanation": explanation,
        "verdict": verdict
    }


# ---------------------------------------------------------
# Self-test
# ---------------------------------------------------------

if __name__ == "__main__":

    # Two locations within the Agartala study area
    test_locations = {
        "Agartala location 1": (
            23.8500,
            91.2800
        ),
        "Agartala location 2": (
            23.8800,
            91.3200
        )
    }

    for name, (lat, lon) in test_locations.items():

        print(f"\n=== {name} ===")

        result = score_location(
            lat,
            lon
        )

        for key, value in result.items():
            print(
                f"{key:20s}: {value}"
            )