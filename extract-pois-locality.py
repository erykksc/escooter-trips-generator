import osmnx as ox
import geopandas as gpd
import pandas as pd
import uuid
import os

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# 1. Download Berlin localities (admin level 9)

tags_localities = {"boundary": "administrative", "admin_level": "9"}

# Note: features_from_place returns a GeoDataFrame
localities = ox.features_from_place("Berlin, Germany", tags=tags_localities)

# Keep only polygons/multipolygons
localities = localities[localities.geometry.type.isin(["Polygon", "MultiPolygon"])]

# Add UUIDs
localities["locality_id"] = [str(uuid.uuid4()) for _ in range(len(localities))]
localities["name"] = localities["name"].fillna("UNKNOWN")

# Select columns to export
localities_export = localities[["locality_id", "name", "geometry"]]

# Export to GeoJSON
localities_export.to_file("output/berlin-localities.geojson", driver="GeoJSON")

print(f"Exported {len(localities_export)} localities to output/berlin_localities.geojson")

# 2. Download Points of Interest (POIs)

tags_pois = {
    "railway": "station",
    "amenity": ["restaurant", "hospital", "school", "bank", "police"],
    "tourism": ["museum", "attraction", "hotel"],
    "shop": ["supermarket", "bicycle", "bakery", "books"],
    "leisure": ["park", "stadium"],
    "historic": "monument",
}

pois = ox.features_from_place("Berlin, Germany", tags=tags_pois)

# Keep only Point geometries
pois_points = pois[pois.geometry.type == "Point"]

# Prepare POIs DataFrame
pois_df = pd.DataFrame(
    {
        "poi_id": [str(uuid.uuid4()) for _ in range(len(pois_points))],
        "name": pois_points["name"].fillna("UNKNOWN"),
        "category": pois_points.apply(
            lambda row: next(
                (key for key in tags_pois.keys() if pd.notnull(row.get(key))), "unknown"
            ),
            axis=1,
        ),
        "longitude": pois_points.geometry.x,
        "latitude": pois_points.geometry.y,
    }
)

# Export POIs to CSV
pois_df.to_csv("output/berlin_pois.csv", index=False)

print(f"Exported {len(pois_df)} POIs to output/berlin_pois.csv")
