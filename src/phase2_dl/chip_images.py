from pathlib import Path

import numpy as np


# ---------------------------------------------------------
# GeoSense Phase II - Exercise 2.3
# Create Sentinel-2 image chips
# ---------------------------------------------------------

CLEAN_DIR = Path("data/satellite/clean")
CHIP_DIR = Path("data/satellite/chips")

YEARS = [2015, 2023]

CHIP_SIZE = 224
STRIDE = 174
MAX_NAN_FRACTION = 0.20


def create_chips(data, output_dir):
    """
    Create overlapping 224 x 224 image chips.

    Chips with more than 20% NaN values are skipped.
    Remaining NaN values are filled using the mean
    value of the corresponding spectral band.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    bands, height, width = data.shape

    chip_count = 0
    skipped_count = 0

    # Calculate the mean of each band while ignoring NaN values
    band_means = np.nanmean(data.reshape(bands, -1), axis=1)

    # Replace invalid band means with zero, if any
    band_means = np.where(
        np.isfinite(band_means),
        band_means,
        0.0
    )

    # Slide the 224 x 224 window across the image
    for row in range(0, height - CHIP_SIZE + 1, STRIDE):

        for col in range(0, width - CHIP_SIZE + 1, STRIDE):

            chip = data[
                :,
                row:row + CHIP_SIZE,
                col:col + CHIP_SIZE
            ].copy()

            # Calculate percentage of NaN values
            nan_fraction = np.mean(np.isnan(chip))

            # Skip chips with more than 20% NaN
            if nan_fraction > MAX_NAN_FRACTION:
                skipped_count += 1
                continue

            # Fill remaining NaN values with band mean
            for band in range(bands):

                nan_mask = np.isnan(chip[band])

                if np.any(nan_mask):
                    chip[band, nan_mask] = band_means[band]

            # Save chip as .npy
            chip_name = f"chip_{chip_count:05d}.npy"

            np.save(
                output_dir / chip_name,
                chip.astype(np.float32)
            )

            chip_count += 1

    return chip_count, skipped_count


def process_year(year):

    input_file = CLEAN_DIR / f"clean_{year}.npy"
    output_dir = CHIP_DIR / str(year)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Clean input file not found: {input_file}"
        )

    print("\n" + "=" * 60)
    print(f"Creating image chips for {year}")
    print("=" * 60)

    data = np.load(input_file)

    print("Input file:", input_file)
    print("Input shape:", data.shape)
    print("Chip size:", CHIP_SIZE)
    print("Stride:", STRIDE)
    print("Overlap:", CHIP_SIZE - STRIDE, "pixels")
    print("Maximum NaN fraction:", MAX_NAN_FRACTION)

    chip_count, skipped_count = create_chips(
        data,
        output_dir
    )

    print("\nChipping complete.")
    print("Year:", year)
    print("Generated chips:", chip_count)
    print("Skipped chips:", skipped_count)
    print("Output directory:", output_dir)


def main():

    print("GeoSense Phase II - Image Chipping")
    print("Years:", YEARS)

    for year in YEARS:
        process_year(year)

    print("\n" + "=" * 60)
    print("Image chipping completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()