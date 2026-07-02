
import random
import numpy as np
import geopandas as gpd

SURFACE_TO_IRI = {
    "asphalt": 2.5,
    "paved": 3.0,
    "concrete": 2.2,
    "paving_stones": 4.0,
    "compacted": 5.5,
    "gravel": 7.0,
    "dirt": 9.0,
    "unpaved": 8.5,
    "ground": 9.5,
    "sand": 10.0,
}

SMOOTHNESS_TO_FACTOR = {
    "excellent": 0.6,
    "good": 0.85,
    "intermediate": 1.0,
    "bad": 1.4,
    "very_bad": 1.8,
    "horrible": 2.2,
    "very_horrible": 2.6,
    "impassable": 3.0,
}

HIGHWAY_CLASS_IRI_RANGE = {
    "trunk": (3.5, 8.5),
    "trunk_link": (4.0, 8.0),
    "primary": (3.0, 7.0),
    "primary_link": (3.5, 7.0),
    "secondary": (4.0, 8.0),
    "secondary_link": (4.5, 8.0),
    "tertiary": (5.0, 9.0),
    "tertiary_link": (5.0, 9.0),
}

DEFAULT_IRI_RANGE = (4.0, 8.0)  



def assign_iri_and_wear_scores(edges_gdf: gpd.GeoDataFrame, seed: int = 42) -> gpd.GeoDataFrame:

    random.seed(seed)
    np.random.seed(seed)

    gdf = edges_gdf.copy()

    iri_scores = []

    for _, row in gdf.iterrows():
        surface = _normalize_tag(row.get("surface"))
        smoothness = _normalize_tag(row.get("smoothness"))
        highway = _normalize_tag(row.get("highway"))

        iri = _estimate_iri(surface, smoothness, highway)
        iri_scores.append(iri)

    gdf["iri_score"] = iri_scores
    gdf["wear_multiplier"] = gdf["iri_score"].apply(iri_to_wear_multiplier)

    return gdf


def _normalize_tag(value):
   
    if value is None:
        return None
    if isinstance(value, list):
       
        value = value[0] if len(value) > 0 else None
    if value is None:
        return None
    try:
        if value != value:  # NaN check
            return None
    except TypeError:
        pass
    return str(value).lower().strip()


def _estimate_iri(surface, smoothness, highway) -> float:
   
    if surface in SURFACE_TO_IRI:
        base_iri = SURFACE_TO_IRI[surface]
        if smoothness in SMOOTHNESS_TO_FACTOR:
            base_iri *= SMOOTHNESS_TO_FACTOR[smoothness]
        
        base_iri *= np.random.uniform(0.95, 1.05)
        return round(float(np.clip(base_iri, 1.0, 12.0)), 2)
    
    low, high = HIGHWAY_CLASS_IRI_RANGE.get(highway, DEFAULT_IRI_RANGE)
    simulated_iri = np.random.uniform(low, high)
    return round(float(simulated_iri), 2)


def iri_to_wear_multiplier(iri_score: float) -> float:
    
    IRI_MIN, IRI_MAX = 2.0, 10.0
    MULT_MIN, MULT_MAX = 1.0, 2.5

    clipped_iri = np.clip(iri_score, IRI_MIN, IRI_MAX)
    ratio = (clipped_iri - IRI_MIN) / (IRI_MAX - IRI_MIN)
    multiplier = MULT_MIN + ratio * (MULT_MAX - MULT_MIN)

    return round(float(multiplier), 3)


if __name__ == "__main__":
    import os

    edges_path = os.path.join("output_data", "nagpur_raipur_edges.geojson")

    if not os.path.exists(edges_path):
        print(f"Could not find {edges_path}.")
        print("Run routing_engine.py first to generate the edges GeoDataFrame.")
    else:
        print(f"Loading edges from: {edges_path}")
        edges_gdf = gpd.read_file(edges_path)

        scored_gdf = assign_iri_and_wear_scores(edges_gdf)

        out_path = os.path.join("output_data", "nagpur_raipur_edges_scored.geojson")
        scored_gdf.to_file(out_path, driver="GeoJSON")

        print(f"Saved scored edges to: {out_path}")
        print("\nPreview:")
        cols_to_show = [c for c in ["highway", "surface", "smoothness", "iri_score", "wear_multiplier"] if c in scored_gdf.columns]
        print(scored_gdf[cols_to_show].head(10))

        print("\nSummary stats:")
        print(scored_gdf[["iri_score", "wear_multiplier"]].describe())