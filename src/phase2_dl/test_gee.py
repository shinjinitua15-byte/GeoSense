import ee

# Initialise Google Earth Engine
ee.Initialize(project="summer-ranger-448906-k0")

# Agartala Municipal Corporation test point
point = ee.Geometry.Point([91.28, 23.85])

# Sentinel-2 Surface Reflectance collection
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(point)
    .filterDate("2023-01-01", "2023-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
)

# Get the first available image
image = collection.first()

print("Sentinel-2 Image ID:")
print(image.id().getInfo())

print("\nSentinel-2 Band List:")
print(image.bandNames().getInfo())

print("\nNumber of matching images:")
print(collection.size().getInfo())