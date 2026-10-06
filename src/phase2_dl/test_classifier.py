from pprint import pprint

from image_classifier import classify_imagery


# ---------------------------------------------------------
# Changed location
# ---------------------------------------------------------

changed_location = (
    23.825971,
    91.268249
)

# ---------------------------------------------------------
# Unchanged location
# ---------------------------------------------------------

unchanged_location = (
    23.831810,
    91.273639
)


def run_test(name, lat, lon):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    result = classify_imagery(
        lat,
        lon,
        radius_m=500
    )

    print("\nReturned fields:")
    pprint(result)

    print("\nSummary:")
    print("Land cover:", result["land_cover"])
    print("Class ID:", result["class_id"])
    print("Confidence (%):", result["confidence_pct"])
    print("NDVI 2023:", result["ndvi"])
    print("NDWI 2023:", result["ndwi"])
    print("NDVI 2015:", result["ndvi_2015"])
    print("NDVI difference:", result["ndvi_difference"])
    print("Change flag:", result["change_flag"])
    print("Change description:")
    print(result["change_description"])
    print("Inference time (sec):", result["inference_time_sec"])


run_test(
    "CHANGED LOCATION",
    changed_location[0],
    changed_location[1]
)

run_test(
    "UNCHANGED LOCATION",
    unchanged_location[0],
    unchanged_location[1]
)