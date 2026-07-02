
import osmnx as ox
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import requests
import os
import time


START_PLACE = "Nagpur, Maharashtra, India"
END_PLACE = "Raipur, Chhattisgarh, India"

HIGHWAY_FILTER = (
    '["highway"~"trunk|primary|secondary|tertiary'
    '|trunk_link|primary_link|secondary_link|tertiary_link"]'
)

ox.settings.overpass_url = "https://overpass-api.de/api"
ox.settings.overpass_rate_limit = True

ox.settings.requests_timeout = 300

OUTPUT_DIR = "output_data"
NODES_FILE = os.path.join(OUTPUT_DIR, "nagpur_raipur_nodes.geojson")
EDGES_FILE = os.path.join(OUTPUT_DIR, "nagpur_raipur_edges.geojson")
GRAPHML_FILE = os.path.join(OUTPUT_DIR, "nagpur_raipur_graph.graphml")
PLOT_FILE = os.path.join(OUTPUT_DIR, "nagpur_raipur_network.png")


def _get_route_geometry(start_point, end_point):
    
    start_lon, start_lat = start_point[1], start_point[0]
    end_lon, end_lat = end_point[1], end_point[0]

    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}"
        f"?overview=full&geometries=geojson"
    )

    print("Tracing driving route via OSRM...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    coords = data["routes"][0]["geometry"]["coordinates"]  # list of [lon, lat]
    route_line = LineString(coords)

    route_line = route_line.simplify(tolerance=0.001, preserve_topology=False)

    print(f"Route traced: ~{data['routes'][0]['distance'] / 1000:.1f} km "
          f"({len(coords)} points, simplified to {len(route_line.coords)})")
    return route_line


def build_fallback_network(route_line, segment_length_km: float = 2.0):
   
    print("Building fallback network directly from route geometry "
          "(bypassing Overpass API)...")

    coords = list(route_line.coords)  # [(lon, lat), ...]

    total_length_deg = route_line.length
    approx_km_per_deg = 111.0
    total_length_km = total_length_deg * approx_km_per_deg
    n_segments = max(int(total_length_km / segment_length_km), 1)

    sample_points = [
        route_line.interpolate(i / n_segments, normalized=True)
        for i in range(n_segments + 1)
    ]

    node_records = []
    for i, pt in enumerate(sample_points):
        node_records.append({
            "osmid": i,
            "x": pt.x,
            "y": pt.y,
            "geometry": pt,
        })
    nodes_gdf = gpd.GeoDataFrame(node_records, geometry="geometry", crs="EPSG:4326")
    nodes_gdf.set_index("osmid", inplace=True)

    edge_records = []
    for i in range(len(sample_points) - 1):
        p1, p2 = sample_points[i], sample_points[i + 1]
        seg_geom = LineString([p1, p2])
        length_m = seg_geom.length * approx_km_per_deg * 1000

        edge_records.append({
            "u": i,
            "v": i + 1,
            "key": 0,
            "osmid": i,
            "highway": "trunk",   
            "surface": None,      
            "smoothness": None,   
            "length": length_m,
            "geometry": seg_geom,
        })

    edges_gdf = gpd.GeoDataFrame(edge_records, geometry="geometry", crs="EPSG:4326")
    edges_gdf.set_index(["u", "v", "key"], inplace=True)

    print(f"Fallback network built: {len(nodes_gdf)} nodes, "
          f"{len(edges_gdf)} edges (~{segment_length_km}km each).")

    return nodes_gdf, edges_gdf


def download_corridor_graph(start_place: str, end_place: str, buffer_km: float = 2.5):
   
    print(f"Geocoding '{start_place}' and '{end_place}'...")
    start_point = ox.geocode(start_place)   # (lat, lon)
    end_point = ox.geocode(end_place)       # (lat, lon)

    route_line = _get_route_geometry(start_point, end_point)

    buffer_deg = buffer_km / 111.0
    corridor_polygon = route_line.buffer(buffer_deg)

    print(f"Downloading road network within {buffer_km}km of the route "
          f"(this may take a minute)...")

    max_retries = 2
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}: sending request to "
              f"{ox.settings.overpass_url} ...")
        attempt_start = time.time()
        try:
            graph = ox.graph_from_polygon(
                corridor_polygon,
                network_type="drive",
                custom_filter=HIGHWAY_FILTER,
                simplify=True,
            )
            elapsed = time.time() - attempt_start
            print(f"Success after {elapsed:.0f}s.")
            print(f"Graph downloaded: {len(graph.nodes)} nodes, "
                  f"{len(graph.edges)} edges.")
            return graph, None
        except Exception as e:
            elapsed = time.time() - attempt_start
            print(f"Attempt {attempt}/{max_retries} failed after {elapsed:.0f}s: {e}")
            if attempt < max_retries:
                wait_seconds = 10 * attempt
                print(f"Retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)

    print("\nLive Overpass download failed after all retries.")
    print("Falling back to building the network directly from route geometry.")
    print("(You can retry the live download later by re-running this script;")
    print(" it will attempt the real OSM data again first.)\n")
    nodes_gdf, edges_gdf = build_fallback_network(route_line)
    return None, (nodes_gdf, edges_gdf)


def save_graph_as_geodataframes(graph=None, fallback_gdfs=None):
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if graph is not None:
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)
        ox.save_graphml(graph, filepath=GRAPHML_FILE)
        print(f"Saved raw graph to: {GRAPHML_FILE}")
    else:
        nodes_gdf, edges_gdf = fallback_gdfs

    nodes_gdf.to_file(NODES_FILE, driver="GeoJSON")
    edges_gdf.to_file(EDGES_FILE, driver="GeoJSON")

    print(f"Saved nodes to: {NODES_FILE}")
    print(f"Saved edges to: {EDGES_FILE}")

    return nodes_gdf, edges_gdf


def visualize_network(edges_gdf, save_path: str = PLOT_FILE, show: bool = True):
    
    fig, ax = plt.subplots(figsize=(10, 8))
    edges_gdf.plot(ax=ax, linewidth=0.9, color="#2c7fb8")
    ax.set_title("Nagpur - Raipur Road Network (Trunk/Primary/Secondary/Tertiary)")
    ax.set_axis_off()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved network plot to: {save_path}")

    if show:
        plt.show()

    plt.close(fig)



if __name__ == "__main__":
    graph, fallback_gdfs = download_corridor_graph(START_PLACE, END_PLACE)
    nodes_gdf, edges_gdf = save_graph_as_geodataframes(graph, fallback_gdfs)
    visualize_network(edges_gdf)

    print("\nDone. Preview of edges GeoDataFrame:")
    print(edges_gdf.head())

    if graph is None:
        print("\nNOTE: this run used the OFFLINE FALLBACK network (Overpass "
              "API was unreachable). Data covers the real route corridor "
              "but without full OSM tag detail. Re-run later to try the "
              "live download again if time permits.")