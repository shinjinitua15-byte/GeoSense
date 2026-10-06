import json
from pathlib import Path

import numpy as np
import rasterio


# ---------------------------------------------------------
# GeoSense Phase II - Exercise 2.2
# Sentinel-2 preprocessing
# ---------------------------------------------------------

RAW_DIR = Path("data/satellite/raw")
CLEAN_DIR = Path("data/satellite/clean")

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2015, 2023]

# Band order:
# B2, B3, B4, B8, B11, B12
B3 = 1
B4 = 2
B8 = 3


def remove_clouds(data):
    """
    Flag pixels as cloud when ALL bands exceed
    reflectance 3000 and replace those pixels with NaN.
    """

    cloud_mask = np.all(data > 3000, axis=0)

    cleaned = data.astype(np.float32)
    cleaned[:, cloud_mask] = np.nan

    return cleaned, cloud_mask


def normalise_to_01(data):
    """
    Divide reflectance values by 10000
    and clip the result to the range 0-1.
    """

    data = data / 10000.0
    data = np.clip(data, 0.0, 1.0)

    return data


def compute_ndvi(data):
    """
    NDVI = (B8 - B4) / (B8 + B4)
    """

    nir = data[B8]
    red = data[B4]

    denominator = nir + red

    ndvi = np.divide(
        nir - red,
        denominator,
        out=np.full_like(nir, np.nan, dtype=np.float32),
        where=denominator != 0
    )

    return ndvi


def compute_ndwi(data):
    """
    NDWI = (B3 - B8) / (B3 + B8)
    """

    green = data[B3]
    nir = data[B8]

    denominator = green + nir

    ndwi = np.divide(
        green - nir,
        denominator,
        out=np.full_like(green, np.nan, dtype=np.float32),
        where=denominator != 0
    )

    return ndwi


def process_file(year):
    """
    Process one Sentinel-2 GeoTIFF.
    """

    input_file = RAW_DIR / f"sentinel2_AMC_{year}.tif"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    print("\n" + "=" * 60)
    print(f"Processing Sentinel-2 {year}")
    print("=" * 60)

    with rasterio.open(input_file) as src:

        data = src.read()

        print("Input file:", input_file)
        print("Input shape:", data.shape)
        print("Data type:", data.dtype)

        # -------------------------------------------------
        # Step 1: Remove clouds
        # -------------------------------------------------

        cleaned_data, cloud_mask = remove_clouds(data)

        total_pixels = cloud_mask.size
        cloud_pixels = int(np.sum(cloud_mask))

        cloud_free_percentage = (
            (total_pixels - cloud_pixels) / total_pixels
        ) * 100.0

        # -------------------------------------------------
        # Step 2: Normalize reflectance
        # -------------------------------------------------

        normalized_data = normalise_to_01(cleaned_data)

        # -------------------------------------------------
        # Step 3: Calculate NDVI and NDWI
        # -------------------------------------------------

        ndvi = compute_ndvi(normalized_data)
        ndwi = compute_ndwi(normalized_data)

        # -------------------------------------------------
        # Step 4: Save output arrays
        # -------------------------------------------------

        clean_file = CLEAN_DIR / f"clean_{year}.npy"
        ndvi_file = CLEAN_DIR / f"ndvi_{year}.npy"
        ndwi_file = CLEAN_DIR / f"ndwi_{year}.npy"
        meta_file = CLEAN_DIR / f"meta_{year}.json"

        np.save(
            clean_file,
            normalized_data.astype(np.float32)
        )

        np.save(
            ndvi_file,
            ndvi.astype(np.float32)
        )

        np.save(
            ndwi_file,
            ndwi.astype(np.float32)
        )

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        metadata = {
            "year": year,
            "source_file": str(input_file),
            "bands": [
                "B2",
                "B3",
                "B4",
                "B8",
                "B11",
                "B12"
            ],
            "input_shape": list(data.shape),
            "normalized_range": [
                0.0,
                1.0
            ],
            "ndvi_mean": float(np.nanmean(ndvi)),
            "ndvi_min": float(np.nanmin(ndvi)),
            "ndvi_max": float(np.nanmax(ndvi)),
            "ndwi_mean": float(np.nanmean(ndwi)),
            "ndwi_min": float(np.nanmin(ndwi)),
            "ndwi_max": float(np.nanmax(ndwi)),
            "cloud_free_percentage": float(
                cloud_free_percentage
            ),
            "cloud_pixels": cloud_pixels,
            "total_pixels": int(total_pixels)
        }

        with open(
            meta_file,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                metadata,
                f,
                indent=4
            )

        # -------------------------------------------------
        # Print results
        # -------------------------------------------------

        print("\nPreprocessing complete.")

        print(
            "Cloud-free percentage:",
            f"{cloud_free_percentage:.2f}%"
        )

        print(
            "NDVI mean:",
            f"{np.nanmean(ndvi):.4f}"
        )

        print(
            "NDVI range:",
            f"{np.nanmin(ndvi):.4f}",
            "to",
            f"{np.nanmax(ndvi):.4f}"
        )

        print(
            "NDWI mean:",
            f"{np.nanmean(ndwi):.4f}"
        )

        print(
            "NDWI range:",
            f"{np.nanmin(ndwi):.4f}",
            "to",
            f"{np.nanmax(ndwi):.4f}"
        )

        print("\nSaved files:")
        print(clean_file)
        print(ndvi_file)
        print(ndwi_file)
        print(meta_file)


def main():

    print("GeoSense Phase II - Sentinel-2 Preprocessing")
    print("Years:", YEARS)

    for year in YEARS:
        process_file(year)

    print("\n" + "=" * 60)
    print("All preprocessing completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()