import os
import geopandas as gpd

from db_connection import get_engine


# Folder containing the downloaded OSM Shapefiles
DATA_DIR = "data/shapefiles/northeastern_osm/"


# OSM layers to be imported into PostGIS
LAYERS_TO_LOAD = [
    ("gis_osm_roads_free_1.shp", "osm_roads"),
    ("gis_osm_buildings_a_free_1.shp", "osm_buildings"),
    ("gis_osm_landuse_a_free_1.shp", "osm_landuse"),
    ("gis_osm_pois_free_1.shp", "osm_poi"),
    ("gis_osm_pois_a_free_1.shp", "osm_poi_area"),
]


def load_layer(filename, table_name, engine):
    """Read one Shapefile and load it into PostGIS."""

    filepath = os.path.join(DATA_DIR, filename)

    print(f"Loading {filename} ...")

    gdf = gpd.read_file(filepath)

    # Keep all layers in WGS 84
    gdf = gdf.to_crs(epsg=4326)

    # Write the layer to PostGIS
    gdf.to_postgis(
        table_name,
        engine,
        if_exists="replace",
        index=False
    )

    print(
        f"Loaded {len(gdf)} features "
        f"into table: {table_name}"
    )


def main():

    engine = get_engine()

    for filename, table_name in LAYERS_TO_LOAD:
        load_layer(
            filename,
            table_name,
            engine
        )

    print("All OSM layers loaded successfully!")


if __name__ == "__main__":
    main()