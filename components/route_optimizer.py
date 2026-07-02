
import os
import json
import requests
import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np

ORS_API_KEY = os.environ.get("ORS_API_KEY", "")

NAGPUR_COORDS = (79.0882, 21.1458)   # (lon, lat)
RAIPUR_COORDS = (81.6296, 21.2514)   # (lon, lat)

SCORED_EDGES_FILE = os.path.join("output_data", "nagpur_raipur_edges_scored.geojson")
OUTPUT_FILE = os.path.join("output_data", "route_comparison.json")

APPROX_KM_PER_DEG = 111.0


def fetch_routes_ors(start_coords, end_coords, api_key, n_alternatives=3):
    
    url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }
    body = {
        "coordinates": [list(start_coords), list(end_coords)],
        "alternative_routes": {
            "target_count": n_alternatives,
            "share_factor": 0.6,   
            "weight_factor": 1.4,  
        },
    }

    print("Requesting alternative routes from OpenRouteService...")
    response = requests.post(url, headers=headers, json=body, timeout=30)
    response.raise_for_status()
    data = response.json()

    routes = []
    for feature in data["features"]:
        coords = feature["geometry"]["coordinates"]  # [lon, lat]
        summary = feature["properties"]["summary"]
        routes.append({
            "geometry": LineString(coords),
            "distance_km": summary["distance"] / 1000,
            "duration_min": summary["duration"] / 60,
        })

    print(f"ORS returned {len(routes)} route(s).")
    return routes


def fetch_routes_osrm(start_coords, end_coords, n_alternatives=3):
    start_lon, start_lat = start_coords
    end_lon, end_lat = end_coords

    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}"
        f"?overview=full&geometries=geojson&alternatives=true"
    )

    print("Requesting alternative routes from OSRM (fallback, no API key)...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    routes = []
    for route in data["routes"][:n_alternatives]:
        coords = route["geometry"]["coordinates"]
        routes.append({
            "geometry": LineString(coords),
            "distance_km": route["distance"] / 1000,
            "duration_min": route["duration"] / 60,
        })

    print(f"OSRM returned {len(routes)} route(s).")
    return routes


def get_alternative_routes(start_coords, end_coords, n_alternatives=3):
    
    if ORS_API_KEY:
        try:
            return fetch_routes_ors(start_coords, end_coords, ORS_API_KEY, n_alternatives)
        except Exception as e:
            print(f"ORS request failed ({e}); falling back to OSRM.")
    else:
        print("No ORS_API_KEY set; using OSRM instead.")

    return fetch_routes_osrm(start_coords, end_coords, n_alternatives)


def load_scored_edges(path=SCORED_EDGES_FILE):
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Run iri_scoring.py first."
        )
    return gpd.read_file(path)


def calculate_wear_cost(route_geometry, edges_gdf, sample_interval_km=1.0):
   
    total_length_km = route_geometry.length * APPROX_KM_PER_DEG
    n_samples = max(int(total_length_km / sample_interval_km), 2)

    sample_points = [
        route_geometry.interpolate(i / n_samples, normalized=True)
        for i in range(n_samples + 1)
    ]

    edge_centroids = edges_gdf.geometry.centroid
    wear_multipliers = edges_gdf["wear_multiplier"].to_numpy()
    centroid_coords = np.array([[pt.x, pt.y] for pt in edge_centroids])

    total_wear_cost = 0.0
    segment_km = total_length_km / n_samples

    for pt in sample_points:
        pt_coord = np.array([pt.x, pt.y])
        distances = np.linalg.norm(centroid_coords - pt_coord, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_wear = wear_multipliers[nearest_idx]
        total_wear_cost += segment_km * nearest_wear

    return round(total_wear_cost, 2)

def compare_routes(routes, edges_gdf):
    comparison = {}

    for i, route in enumerate(routes, start=1):
        label = f"Route {i}"
        wear_cost = calculate_wear_cost(route["geometry"], edges_gdf)

        comparison[label] = {
            "distance_km": round(route["distance_km"], 1),
            "duration_min": round(route["duration_min"], 1),
            "wear_cost": wear_cost,
        }

    ranking = sorted(comparison.keys(), key=lambda k: comparison[k]["wear_cost"])

    result = {
        "routes": comparison,
        "ranking_by_wear_cost": ranking,
        "recommended_route": ranking[0],
    }
    return result


if __name__ == "__main__":
    routes = get_alternative_routes(NAGPUR_COORDS, RAIPUR_COORDS, n_alternatives=3)
    edges_gdf = load_scored_edges()

    result = compare_routes(routes, edges_gdf)

    os.makedirs("output_data", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved route comparison to: {OUTPUT_FILE}\n")
    print(json.dumps(result, indent=2))