
import csv
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------
PROJECT_ROOT = Path(r"D:\GeoSense_Phase3")

ENV_FILE = PROJECT_ROOT / ".env"
IMAGE_DIR = PROJECT_ROOT / "data" / "streetview" / "images"
RESULTS_DIR = PROJECT_ROOT / "data" / "streetview" / "results"
METADATA_FILE = RESULTS_DIR / "mapillary_image_metadata.csv"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ENV_FILE)

ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN", "").strip()

if not ACCESS_TOKEN:
    raise RuntimeError(
        "MAPILLARY_ACCESS_TOKEN was not found in "
        f"{ENV_FILE}. Add your replacement token to the .env file."
    )


# --------------------------------------------------
# 2. MAPILLARY SETTINGS
# --------------------------------------------------
API_URL = "https://graph.mapillary.com/images"

FIELDS = (
    "id,thumb_2048_url,geometry,"
    "compass_angle,captured_at"
)

MAX_IMAGES_PER_AREA = 20
REQUEST_TIMEOUT = 45

# Bounding boxes use:
# minimum longitude, minimum latitude,
# maximum longitude, maximum latitude.
AREAS = [
    {
        "name": "Original study area",
        "bbox": (80.24, 13.05, 80.25, 13.06),
    },
    {
        "name": "Chennai city centre",
        "bbox": (80.265, 13.077, 80.277, 13.089),
    },
]


# --------------------------------------------------
# 3. HTTP SESSION
# --------------------------------------------------
session = requests.Session()
session.headers.update(
    {"User-Agent": "GeoSense-Phase3-StreetView/1.0"}
)


# --------------------------------------------------
# 4. LOAD EXISTING METADATA
# --------------------------------------------------
def load_existing_metadata():
    records = {}

    if not METADATA_FILE.exists():
        return records

    with METADATA_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_id = row.get("id", "").strip()
            if image_id:
                records[image_id] = row

    return records


# --------------------------------------------------
# 5. QUERY MAPILLARY
# --------------------------------------------------
def search_area(area_name, bbox):
    print(f"\nSearching: {area_name}")
    print("Bounding box:", ",".join(map(str, bbox)))

    params = {
        "access_token": ACCESS_TOKEN,
        "fields": FIELDS,
        "bbox": ",".join(map(str, bbox)),
        "limit": 20,
    }

    found = []
    next_url = API_URL
    first_request = True

    try:
        while len(found) < MAX_IMAGES_PER_AREA:
            if first_request:
                response = session.get(
                    next_url,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                first_request = False
            else:
                # Mapillary may return a pagination URL.
                # Do not print it because it may contain credentials.
                response = session.get(
                    next_url,
                    timeout=REQUEST_TIMEOUT,
                )

            if response.status_code == 401:
                print(
                    "Authentication failed. Check that the "
                    "replacement Mapillary token is valid."
                )
                break

            if response.status_code == 403:
                print(
                    "Access denied. Check the token permissions "
                    "and Mapillary API access."
                )
                break

            if not response.ok:
                print(
                    f"Mapillary returned HTTP "
                    f"{response.status_code}."
                )
                break

            payload = response.json()

            if "error" in payload:
                error = payload["error"]
                print(
                    "Mapillary API error:",
                    error.get("message", "Unknown error"),
                )
                break

            batch = payload.get("data", [])
            if not batch:
                print("No more images returned for this area.")
                break

            found.extend(batch)

            paging = payload.get("paging", {})
            next_url = paging.get("next")

            if not next_url:
                break

            time.sleep(0.3)

    except requests.exceptions.RequestException as exc:
        # Avoid printing request URLs, which can expose tokens.
        print(
            "Network request failed. Check your internet/DNS "
            "connection and try again."
        )
        print("Error type:", type(exc).__name__)

    except ValueError:
        print("Mapillary returned a response that was not valid JSON.")

    print(f"Images found in this search: {len(found)}")
    return found


# --------------------------------------------------
# 6. CONVERT API RECORDS TO CSV ROWS
# --------------------------------------------------
def make_metadata_row(item, area_name):
    geometry = item.get("geometry") or {}
    coordinates = geometry.get("coordinates") or [None, None]

    # Mapillary GeoJSON coordinates are [longitude, latitude].
    longitude = coordinates[0] if len(coordinates) > 0 else None
    latitude = coordinates[1] if len(coordinates) > 1 else None

    image_id = str(item.get("id", "")).strip()

    return {
        "id": image_id,
        "filename": f"{image_id}.jpg",
        "latitude": latitude,
        "longitude": longitude,
        "thumb_2048_url": item.get("thumb_2048_url", ""),
        "compass_angle": item.get("compass_angle", ""),
        "captured_at": item.get("captured_at", ""),
        "area": area_name,
    }


# --------------------------------------------------
# 7. DOWNLOAD IMAGES
# --------------------------------------------------
def download_image(row):
    image_id = row["id"]
    filename = row["filename"]
    image_path = IMAGE_DIR / filename
    image_url = row.get("thumb_2048_url", "")

    # Preserve valid images already downloaded.
    if image_path.exists() and image_path.stat().st_size > 0:
        return "Already exists"

    if not image_url:
        return "No image URL"

    try:
        response = session.get(
            image_url,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if "image" not in content_type:
            return "Response was not an image"

        if not response.content:
            return "Empty image response"

        image_path.write_bytes(response.content)
        return "Downloaded"

    except requests.exceptions.RequestException:
        # Do not print the image URL or its query parameters.
        return "Download failed (network or HTTP error)"


# --------------------------------------------------
# 8. SAVE MERGED METADATA
# --------------------------------------------------
CSV_FIELDS = [
    "id",
    "filename",
    "latitude",
    "longitude",
    "thumb_2048_url",
    "compass_angle",
    "captured_at",
    "area",
]


def save_metadata(records):
    with METADATA_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()

        for image_id in sorted(records):
            writer.writerow(records[image_id])


# --------------------------------------------------
# 9. MAIN WORKFLOW
# --------------------------------------------------
def main():
    records = load_existing_metadata()

    print("GeoSense Phase III — Mapillary downloader")
    print("Existing metadata records:", len(records))
    print("Existing image files will be preserved.")
    print("Access token is loaded from .env (not displayed).")

    for area in AREAS:
        results = search_area(
            area["name"],
            area["bbox"],
        )

        for item in results:
            row = make_metadata_row(item, area["name"])

            if not row["id"]:
                continue

            # Preserve useful existing fields if a new response
            # omits any optional metadata.
            previous = records.get(row["id"], {})
            merged = dict(previous)
            merged.update({
                key: value
                for key, value in row.items()
                if value not in (None, "")
            })
            records[row["id"]] = merged

    # Save metadata before downloading, so discovered IDs are kept
    # even if a later image download fails.
    save_metadata(records)

    downloaded = 0
    already_exists = 0
    failed = 0

    for row in records.values():
        status = download_image(row)

        if status == "Downloaded":
            downloaded += 1
            print("Downloaded:", row["filename"])
        elif status == "Already exists":
            already_exists += 1
        else:
            failed += 1
            print(row["filename"], ":", status)

        time.sleep(0.15)

    print("\nProcess finished.")
    print("Metadata records:", len(records))
    print("New downloads:", downloaded)
    print("Existing images preserved:", already_exists)
    print("Images not downloaded:", failed)
    print("Metadata CSV:", METADATA_FILE)
    print("Image folder:", IMAGE_DIR)


if __name__ == "__main__":
    main()
