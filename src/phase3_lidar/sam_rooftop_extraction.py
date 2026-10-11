
from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------
PROJECT_ROOT = Path(r"D:\GeoSense_Phase3")

IMAGE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "aerial"
    / "boise_rooftop_crop.png"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "saved" / "sam"
    / "sam_vit_b_01ec64.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sam_rooftops_boise"
ROOF_DIR = OUTPUT_DIR / "selected_rooftops"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROOF_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_SIZE_M = 0.6
PIXEL_AREA_M2 = PIXEL_SIZE_M ** 2
AVERAGE_BUILDING_HEIGHT_M = 8.4

MIN_MASK_AREA_PX = 40
MAX_MASK_AREA_PX = 4000

# Only the two manually selected rooftop segments.
SELECTED_SEGMENT_RANKS = [6, 7]


# --------------------------------------------------
# 2. LOAD IMAGE AND SAM MODEL
# --------------------------------------------------
def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Aerial image not found: {IMAGE_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {CHECKPOINT_PATH}"
        )

    image = np.array(
        Image.open(IMAGE_PATH).convert("RGB")
    )

    print("Input image:", IMAGE_PATH)
    print("Image dimensions:", image.shape[1], "x", image.shape[0])
    print("Pixel resolution:", PIXEL_SIZE_M, "m")
    print("Pixel area:", PIXEL_AREA_M2, "m²")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Processing device:", device)

    sam = sam_model_registry["vit_b"](
        checkpoint=str(CHECKPOINT_PATH)
    )
    sam.to(device=device)

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=24,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        crop_n_layers=0,
        min_mask_region_area=MIN_MASK_AREA_PX,
    )

    # --------------------------------------------------
    # 3. GENERATE AND RANK MASKS
    # --------------------------------------------------
    print("\nGenerating segmentation masks...")
    masks = generator.generate(image)

    candidates = [
        mask for mask in masks
        if MIN_MASK_AREA_PX <= mask["area"] <= MAX_MASK_AREA_PX
    ]

    # Rank by area, largest first.
    ranked_masks = sorted(
        candidates,
        key=lambda mask: mask["area"],
        reverse=True
    )

    print("Total masks generated:", len(masks))
    print("Candidates retained:", len(candidates))

    # Ranks 6 and 7 refer to the sorted segments from the
    # earlier run. Recheck their identity in the new run,
    # because SAM mask rankings may change between runs.
    selected = []

    for rank in SELECTED_SEGMENT_RANKS:
        if rank <= len(ranked_masks):
            selected.append((rank, ranked_masks[rank - 1]))

    if len(selected) != 2:
        raise RuntimeError(
            "Could not find both selected segments 6 and 7."
        )

    # --------------------------------------------------
    # 4. SAVE ONLY THE TWO SELECTED ROOFTOP CANDIDATES
    # --------------------------------------------------
    csv_path = OUTPUT_DIR / "selected_rooftops_only.csv"
    overlay = image.copy()

    rows = []

    colors = [
        np.array([255, 40, 40], dtype=np.uint8),
        np.array([40, 220, 255], dtype=np.uint8),
    ]

    for (rank, mask), color in zip(selected, colors):
        segmentation = mask["segmentation"]
        area_pixels = int(mask["area"])
        footprint_m2 = area_pixels * PIXEL_AREA_M2
        volume_m3 = footprint_m2 * AVERAGE_BUILDING_HEIGHT_M

        # Save binary mask.
        binary = np.zeros(segmentation.shape, dtype=np.uint8)
        binary[segmentation] = 255

        mask_path = ROOF_DIR / f"rooftop_segment_{rank}.png"
        Image.fromarray(binary).save(mask_path)

        # Save the outline on the original aerial image.
        outline = image.copy()

        # Find the mask boundary using neighbouring pixels.
        padded = np.pad(
            segmentation,
            1,
            mode="constant",
            constant_values=False
        )

        interior = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )

        boundary = segmentation & ~interior
        outline[boundary] = color

        outline_path = ROOF_DIR / f"rooftop_segment_{rank}_outline.png"
        Image.fromarray(outline).save(outline_path)

        # Add selected segment to the combined overlay.
        overlay[segmentation] = (
            0.5 * overlay[segmentation].astype(np.float32)
            + 0.5 * color.astype(np.float32)
        ).astype(np.uint8)

        rows.append({
            "segment_rank": rank,
            "area_pixels": area_pixels,
            "pixel_area_m2": PIXEL_AREA_M2,
            "estimated_footprint_m2": round(footprint_m2, 2),
            "assumed_height_m": AVERAGE_BUILDING_HEIGHT_M,
            "preliminary_volume_m3": round(volume_m3, 2),
            "predicted_iou": round(
                float(mask.get("predicted_iou", 0)), 4
            ),
            "stability_score": round(
                float(mask.get("stability_score", 0)), 4
            ),
            "verification_status": "Manually selected; roof boundary requires verification"
        })

        print(f"\nSelected segment {rank}")
        print("Area:", area_pixels, "pixels")
        print("Estimated footprint:", round(footprint_m2, 2), "m²")
        print("Assumed height:", AVERAGE_BUILDING_HEIGHT_M, "m")
        print("Preliminary volume:", round(volume_m3, 2), "m³")
        print("Mask saved:", mask_path)
        print("Outline saved:", outline_path)

    # Save selected-rooftop table.
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Save overlay containing only the two selected segments.
    overlay_path = OUTPUT_DIR / "selected_rooftops_overlay.png"
    Image.fromarray(overlay).save(overlay_path)

    total_area = sum(row["estimated_footprint_m2"] for row in rows)
    total_volume = sum(row["preliminary_volume_m3"] for row in rows)

    print("\n" + "=" * 45)
    print("SELECTED ROOFTOP CANDIDATES SUMMARY")
    print("=" * 45)
    print("Selected segments:", len(rows))
    print("Combined estimated footprint:", round(total_area, 2), "m²")
    print("Combined preliminary volume:", round(total_volume, 2), "m³")
    print("CSV:", csv_path)
    print("Overlay:", overlay_path)
    print("Output folder:", ROOF_DIR)
    print("\nImportant: confirm roof boundaries before treating")
    print("these measurements as actual building footprints.")

    print("\nProcessing completed.")


if __name__ == "__main__":
    main()
