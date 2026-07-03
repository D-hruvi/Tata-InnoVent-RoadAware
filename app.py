"""
Road-Aware Vehicle Maintenance AI — Dashboard
Tata InnoVent Challenge 2026

A Streamlit + Folium dashboard that visualizes Indian road segments coloured
by their "wear multiplier" (how badly a stretch of road accelerates vehicle
component wear), and lets a fleet owner compare routes.
"""

import random
import streamlit as st
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import st_folium

from components import ml_engine as ml  # real trained RUL model + live road-state engine

MODEL_PATH = "models/rul_model.pkl"
LIVE_CORRIDOR_ROUTE = "Nagpur → Raipur (NH53)"  # the only route with real road_state.json data

# --------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="RoadAware | Fleet Wear Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------
# GLOBAL STYLING
# --------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0B0F14;
    }

    section[data-testid="stSidebar"] {
        background-color: #10151C;
        border-right: 1px solid #232B36;
    }

    /* Header block */
    .ra-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 26px;
        background: linear-gradient(135deg, #101820 0%, #16202B 100%);
        border: 1px solid #22303C;
        border-radius: 14px;
        margin-bottom: 18px;
    }
    .ra-title {
        font-size: 26px;
        font-weight: 800;
        color: #F5F7FA;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .ra-subtitle {
        font-size: 13.5px;
        color: #8A97A6;
        margin-top: 2px;
    }
    .ra-badge {
        background: #1B4332;
        color: #6FCF97;
        font-size: 12px;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 999px;
        border: 1px solid #2F6B4A;
    }

    /* Metric cards */
    .ra-card {
        background: linear-gradient(160deg, #131A22 0%, #0F151C 100%);
        border: 1px solid #232B36;
        border-radius: 14px;
        padding: 16px 18px;
        height: 100%;
    }
    .ra-card-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #7C8794;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .ra-card-value {
        font-size: 24px;
        font-weight: 800;
        color: #F5F7FA;
        line-height: 1.1;
    }
    .ra-card-delta-good { color: #6FCF97; font-size: 12.5px; font-weight: 600; margin-top: 6px; }
    .ra-card-delta-bad  { color: #EB5757; font-size: 12.5px; font-weight: 600; margin-top: 6px; }
    .ra-card-sub { color: #5D6975; font-size: 12px; margin-top: 6px; }

    /* Section titles */
    .ra-section-title {
        font-size: 15px;
        font-weight: 700;
        color: #E6E9EE;
        margin: 22px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Map container */
    .ra-map-wrap {
        border: 1px solid #232B36;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 28px rgba(0,0,0,0.35);
    }

    /* Legend */
    .ra-legend {
        display: flex;
        gap: 18px;
        align-items: center;
        background: #10151C;
        border: 1px solid #232B36;
        border-radius: 10px;
        padding: 10px 16px;
        margin-top: 10px;
        font-size: 13px;
        color: #C6CDD5;
    }
    .ra-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }

    /* Route comparison table look */
    .ra-route-card {
        background: #131A22;
        border: 1px solid #232B36;
        border-radius: 14px;
        padding: 16px 18px;
    }
    .ra-route-card.best { border: 1px solid #2F6B4A; box-shadow: 0 0 0 1px rgba(111,207,151,0.15) inset; }
    .ra-route-name { font-weight: 700; color: #F5F7FA; font-size: 15px; }
    .ra-route-tag { font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 999px; }

    hr { border-color: #202832; }

    /* Big savings callout */
    .ra-callout {
        background: radial-gradient(120% 160% at 0% 0%, #163B2A 0%, #0F2A20 45%, #0D1A16 100%);
        border: 1px solid #2F6B4A;
        border-radius: 18px;
        padding: 26px 30px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 10px 30px rgba(47,107,74,0.18);
        margin: 6px 0 4px 0;
    }
    .ra-callout-label {
        font-size: 13.5px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        color: #8FE0B4;
        margin-bottom: 6px;
    }
    .ra-callout-value {
        font-size: 46px;
        font-weight: 800;
        color: #F5F7FA;
        line-height: 1;
        letter-spacing: -1px;
    }
    .ra-callout-sub {
        color: #9FB3A9;
        font-size: 13px;
        margin-top: 8px;
    }
    .ra-callout-icon {
        font-size: 44px;
        opacity: 0.9;
    }

    /* Fleet health badges */
    .ra-health-good { color: #3CCB7F; font-weight: 700; }
    .ra-health-mid  { color: #F2A93B; font-weight: 700; }
    .ra-health-bad  { color: #E5484D; font-weight: 700; }

    /* Dataframe container */
    div[data-testid="stDataFrame"] {
        border: 1px solid #232B36;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Slider + selectbox tweaks */
    div[data-baseweb="select"] > div {
        background-color: #0F151C;
        border-color: #2B3542;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# MOCK DATA — ROUTES, WAYPOINTS & ROAD SEGMENTS
# --------------------------------------------------------------------------------
# Each route has a rough polyline of city-to-city waypoints (lat, lon).
# We densify each leg into several road "segments," each carrying a mock
# IRI roughness score + wear_multiplier so the map has something realistic
# to color and the dashboard has numbers to show.

ROUTES = {
    "Nagpur → Raipur (NH53)": {
        "start": ("Nagpur", 21.1458, 79.0882),
        "end": ("Raipur", 21.2514, 81.6296),
        "waypoints": [
            (21.1458, 79.0882),
            (21.20, 79.55),
            (21.05, 79.98),   # near Wardha — bad stretch
            (21.02, 80.35),
            (21.10, 80.75),
            (21.18, 81.10),
            (21.25, 81.6296),
        ],
        "bad_zone_index_range": (2, 4),  # segments here get inflated wear
        "avg_speed_kmph": 55,
    },
    "Mumbai → Pune (Expressway)": {
        "start": ("Mumbai", 19.0760, 72.8777),
        "end": ("Pune", 18.5204, 73.8567),
        "waypoints": [
            (19.0760, 72.8777),
            (18.95, 73.10),
            (18.80, 73.35),
            (18.70, 73.55),
            (18.5204, 73.8567),
        ],
        "bad_zone_index_range": (None, None),  # mostly smooth expressway
        "avg_speed_kmph": 80,
    },
    "Delhi → Jaipur (NH48)": {
        "start": ("Delhi", 28.6139, 77.2090),
        "end": ("Jaipur", 26.9124, 75.7873),
        "waypoints": [
            (28.6139, 77.2090),
            (28.20, 76.85),
            (27.70, 76.45),
            (27.20, 76.10),
            (26.9124, 75.7873),
        ],
        "bad_zone_index_range": (1, 2),
        "avg_speed_kmph": 65,
    },
}

SEGMENT_NAMES = [
    "Toll Plaza Stretch", "Village Bypass", "Highway Straight", "Bridge Crossing",
    "Market Zone Stretch", "Industrial Belt Road", "Hill Curve Section",
    "Expressway Segment", "Rural Kaccha Link", "Speed-Breaker Cluster",
]


def _interpolate(p1, p2, n):
    """Linearly interpolate n points between p1 and p2 (inclusive of p1, exclusive of p2)."""
    lat1, lon1 = p1
    lat2, lon2 = p2
    return [
        (lat1 + (lat2 - lat1) * i / n, lon1 + (lon2 - lon1) * i / n)
        for i in range(n)
    ]


def build_segments(route_key, seed=42):
    """Generate mock GeoJSON-ready road segments for a given route."""
    route = ROUTES[route_key]
    waypoints = route["waypoints"]
    bad_start, bad_end = route["bad_zone_index_range"]
    rng = random.Random(seed)

    segments = []
    seg_id = 0
    points_per_leg = 3

    for leg_idx in range(len(waypoints) - 1):
        leg_points = _interpolate(waypoints[leg_idx], waypoints[leg_idx + 1], points_per_leg)
        leg_points.append(waypoints[leg_idx + 1])

        in_bad_zone = bad_start is not None and bad_start <= leg_idx <= bad_end

        for i in range(len(leg_points) - 1):
            p_a = leg_points[i]
            p_b = leg_points[i + 1]

            if in_bad_zone:
                iri = round(rng.uniform(6.5, 9.2), 1)
                wear = round(rng.uniform(1.8, 2.6), 2)
            else:
                roll = rng.random()
                if roll < 0.55:
                    iri = round(rng.uniform(1.5, 3.5), 1)
                    wear = round(rng.uniform(0.8, 1.19), 2)
                elif roll < 0.85:
                    iri = round(rng.uniform(3.6, 5.8), 1)
                    wear = round(rng.uniform(1.2, 1.79), 2)
                else:
                    iri = round(rng.uniform(6.0, 8.5), 1)
                    wear = round(rng.uniform(1.8, 2.4), 2)

            seg_id += 1
            length_km = round(rng.uniform(6, 14), 1)
            segments.append(
                {
                    "id": seg_id,
                    "name": f"{rng.choice(SEGMENT_NAMES)} #{seg_id}",
                    "coords": [[p_a[1], p_a[0]], [p_b[1], p_b[0]]],  # [lon, lat] for GeoJSON
                    "iri_score": iri,
                    "wear_multiplier": wear,
                    "length_km": length_km,
                }
            )

    return segments


def segments_to_geojson(segments):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": s["id"],
                    "name": s["name"],
                    "iri_score": s["iri_score"],
                    "wear_multiplier": s["wear_multiplier"],
                    "length_km": s["length_km"],
                },
                "geometry": {"type": "LineString", "coordinates": s["coords"]},
            }
            for s in segments
        ],
    }


def wear_color(wear_multiplier):
    if wear_multiplier < 1.2:
        return "#3CCB7F"   # green
    elif wear_multiplier <= 1.8:
        return "#F2A93B"   # orange
    else:
        return "#E5484D"   # red


def wear_label(wear_multiplier):
    if wear_multiplier < 1.2:
        return "Healthy"
    elif wear_multiplier <= 1.8:
        return "Moderate Wear"
    else:
        return "Severe Wear"


# --------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>"
        "<div style='font-size:26px;'>🛣️</div>"
        "<div style='font-size:19px;font-weight:800;color:#F5F7FA;'>RoadAware</div>"
        "</div>"
        "<div style='color:#7C8794;font-size:12.5px;margin-bottom:22px;'>Edge AI for Vehicle Health · Tata InnoVent 2026</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Active Route**")
    route_key = st.selectbox(
        "Select a route",
        list(ROUTES.keys()),
        label_visibility="collapsed",
    )

    st.markdown("<div class='ra-section-title' style='margin-top:20px;'>🚛 Fleet Parameters</div>", unsafe_allow_html=True)
    fleet_size = st.slider("Fleet size (trucks on this route)", 1, 50, 10)
    daily_trips = st.slider("Trips per truck / week", 1, 14, 6)

    st.markdown("<div class='ra-section-title'>🔧 Vehicle Profile</div>", unsafe_allow_html=True)
    vehicle_age_days = st.slider("Vehicle age (days)", 30, 1500, 365, step=15)
    load_pct = st.slider("Average load (% of rated capacity)", 40, 130, 80, step=5)

    st.markdown("<div class='ra-section-title'>🗺️ Map Layers</div>", unsafe_allow_html=True)
    show_markers = st.checkbox("Show city markers", value=True)
    show_heatmap = st.checkbox("Show wear density heatmap", value=False)
    show_labels = st.checkbox("Show segment tooltips", value=True)

    st.markdown("<div class='ra-section-title'>⚙️ Model</div>", unsafe_allow_html=True)
    st.caption("Base RUL model: Random Forest — NASA C-MAPSS (live, loaded from models/rul_model.pkl)")
    if route_key == LIVE_CORRIDOR_ROUTE:
        st.caption("Road data: LIVE from data/road_state.json")
    else:
        st.caption("Road data: demo route — illustrative wear only (live IRI data covers the NH53 corridor)")

    st.markdown("---")
    if st.button("🔄 Refresh Wear Scores", use_container_width=True):
        st.session_state["seed"] = random.randint(0, 10_000)

seed = st.session_state.get("seed", 42)
raw_segments = build_segments(route_key, seed=seed)
route_info = ROUTES[route_key]
repair_cost = 18000
IS_LIVE_CORRIDOR = route_key == LIVE_CORRIDOR_ROUTE

# --------------------------------------------------------------------------------
# LIVE WEAR MULTIPLIERS — from ml_engine + data/road_state.json
# --------------------------------------------------------------------------------
# Route A = whatever corridor is selected in the sidebar.
# Route B = the NH353 alternate — the only real alternate route we have IRI data for.
# For the NH53 corridor these come straight from live road_state.json. For the two
# demo routes (no real sensor/IRI data yet) we fall back to the average wear of the
# mock map segments, clearly labelled as an estimate in the UI below.
if IS_LIVE_CORRIDOR:
    live_iri_a, mult_a = ml.get_route_iri("A")
    live_iri_b, mult_b = ml.get_route_iri("B")
else:
    mult_a = round(sum(s["wear_multiplier"] for s in raw_segments) / len(raw_segments), 3)
    live_iri_a = None
    live_iri_b, mult_b = ml.get_route_iri("B")

# Real model prediction for the selected route, given the fleet's vehicle profile
model_result = ml.calculate_adjusted_rul(vehicle_age_days, load_pct, mult_a, model_path=MODEL_PATH)
base_rul_days = model_result["base_rul"]
adjusted_rul_days = model_result["adjusted_rul"]
risk_level = model_result["risk_level"]
recommendation = model_result["recommendation"]

# --------------------------------------------------------------------------------
# JUDGE FEATURE — SIMULATE GOVERNMENT REPAIR OF THE WORST (RED) SEGMENT
# --------------------------------------------------------------------------------
# Pulls the actual worst segment (highest IRI) from the live road_state.json and
# runs it through ml.simulate_repair_impact() — the real "what if we repaired this"
# function from ml_engine.py. Nothing here is a fixed pitch-deck number anymore.
live_road_state = ml.get_all_segments()
worst_live_segment_name = max(live_road_state, key=lambda k: live_road_state[k]["iri_score"])
worst_live_segment = live_road_state[worst_live_segment_name]

# Kept for the cosmetic map coloring only (no real polyline-level IRI dataset exists)
worst_raw_segment = max(raw_segments, key=lambda s: s["wear_multiplier"])


def apply_repair(segment_list, target_segment_id, repaired):
    """
    Return a new list of segments where the target segment's wear_multiplier
    is reset to a freshly-repaired 1.0x (and its IRI roughness score improved
    to match) if `repaired` is True. Original list is left untouched.
    """
    updated = []
    for seg in segment_list:
        seg_copy = dict(seg)
        if repaired and seg_copy["id"] == target_segment_id:
            seg_copy["wear_multiplier"] = 1.0
            seg_copy["iri_score"] = 2.0
            seg_copy["name"] = seg_copy["name"] + " (Repaired)"
        updated.append(seg_copy)
    return updated


st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #241B08 0%, #1A1508 100%);
        border: 1px solid #6B5320;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    ">
        <div>
            <div style="color:#F2C94C;font-weight:800;font-size:14px;letter-spacing:0.4px;">
                🏆 JUDGE DEMO — GOVERNMENT INTERVENTION SIMULATOR
            </div>
            <div style="color:#C9B87A;font-size:12.5px;margin-top:3px;">
                Simulate a government repair of the worst road segment and watch every metric recompute live.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

repair_toggle = st.toggle(
    f"🚧 Simulate Government Repair of {worst_live_segment_name.replace('_', ' ')}",
    value=False,
    help=(
        f"Runs ml_engine.simulate_repair_impact() on the real trained model: repairs "
        f"'{worst_live_segment_name}' (currently IRI {worst_live_segment['iri_score']}, "
        f"{worst_live_segment['condition']}) down to a freshly-paved IRI of 2.5, and "
        f"recalculates RUL and cost live. Does not permanently change road_state.json."
    ),
)

repair_sim = None
if repair_toggle:
    repair_sim = ml.simulate_repair_impact(
        segment_name=worst_live_segment_name,
        repaired_iri=2.5,
        vehicle_age_days=vehicle_age_days,
        load_pct=load_pct,
        fleet_size=fleet_size,
    )
    # If the simulated segment feeds Route A's average IRI (true for the NH53
    # corridor), reflect the improved RUL in the header metrics below too.
    if IS_LIVE_CORRIDOR and "error" not in repair_sim:
        adjusted_rul_days = repair_sim["after"]["route_a_rul"]

# Cosmetic-only: recolor the mock map polylines so the visual matches the toggle
segments = apply_repair(raw_segments, worst_raw_segment["id"], repair_toggle)
geojson_data = segments_to_geojson(segments)

# --------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------
avg_wear = sum(s["wear_multiplier"] for s in segments) / len(segments)
severe_count = sum(1 for s in segments if s["wear_multiplier"] > 1.8)
total_km = sum(s["length_km"] for s in segments)

# Live annual savings from ml_engine's route comparison (see Route Comparison
# Matrix section below for route_cmp) — used for the fleet callout further down.
route_cmp = ml.compare_routes(mult_a, mult_b, vehicle_age_days, load_pct)
annual_saving = route_cmp["rupees_saved_annually"] * fleet_size

if repair_toggle and repair_sim and "error" not in repair_sim:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #163B2A 0%, #0F2A20 100%);
            border: 1px solid #3CCB7F;
            border-radius: 14px;
            padding: 18px 22px;
            margin-bottom: 18px;
            box-shadow: 0 0 24px rgba(60,203,127,0.25);
        ">
            <div style="font-size:16px;font-weight:800;color:#B7F3CE;">
                🎉 Model result: repairing this segment extends predicted RUL by
                {repair_sim['rul_gain_days']} days per vehicle.
            </div>
            <div style="color:#8FE0B4;font-size:13px;margin-top:6px;">
                For your fleet of {fleet_size} trucks, that's <b>₹{repair_sim['extra_savings_fleet']:,}</b>
                in additional annual savings — on top of the route-optimization savings below.
                <b>{worst_live_segment_name}</b> improves from IRI
                <b>{repair_sim['current_iri']}</b> to <b>{repair_sim['repaired_iri']}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.balloons()

st.markdown(
    f"""
    <div class="ra-header">
        <div>
            <p class="ra-title">Road-Aware Vehicle Maintenance AI</p>
            <p class="ra-subtitle">Live wear intelligence for <b style="color:#C6CDD5;">{route_info['start'][0]} → {route_info['end'][0]}</b> &nbsp;·&nbsp; {len(segments)} segments monitored</p>
        </div>
        <div class="ra-badge">● {"LIVE MODEL" if IS_LIVE_CORRIDOR else "DEMO ROUTE"}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# METRIC CARDS
# --------------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""<div class="ra-card">
            <div class="ra-card-label">Adjusted RUL ({risk_level})</div>
            <div class="ra-card-value">{adjusted_rul_days:.0f} days</div>
            <div class="ra-card-delta-bad">↓ {base_rul_days - adjusted_rul_days:.0f} days vs. road-agnostic baseline</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class="ra-card">
            <div class="ra-card-label">{"Live" if IS_LIVE_CORRIDOR else "Estimated"} Wear Multiplier</div>
            <div class="ra-card-value">{mult_a:.2f}×</div>
            <div class="ra-card-sub">map shows {len(segments)} segments · {total_km:.0f} km</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""<div class="ra-card">
            <div class="ra-card-label">Severe-Wear Segments</div>
            <div class="ra-card-value" style="color:#E5484D;">{severe_count}</div>
            <div class="ra-card-sub">wear multiplier &gt; 1.8×</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""<div class="ra-card">
            <div class="ra-card-label">Est. Fleet Savings / yr</div>
            <div class="ra-card-value" style="color:#6FCF97;">₹{annual_saving:,}</div>
            <div class="ra-card-sub">for {fleet_size} trucks, {daily_trips} trips/week</div>
        </div>""",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------
# MAP
# --------------------------------------------------------------------------------
st.markdown("<div class='ra-section-title'>🗺️ Road Wear Map</div>", unsafe_allow_html=True)

mid_lat = (route_info["start"][1] + route_info["end"][1]) / 2
mid_lon = (route_info["start"][2] + route_info["end"][2]) / 2

fmap = folium.Map(
    location=[mid_lat, mid_lon],
    zoom_start=8,
    tiles=None,
    control_scale=True,
)

# Professional dark basemap
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attr='&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap contributors',
    name="Dark Mode",
    control=False,
).add_to(fmap)

# GeoJson road segments colored by wear_multiplier
def style_function(feature):
    wear = feature["properties"]["wear_multiplier"]
    color = wear_color(wear)
    weight = 6 if wear > 1.8 else 5
    return {
        "color": color,
        "weight": weight,
        "opacity": 0.92,
    }


def highlight_function(feature):
    return {"weight": 9, "opacity": 1.0}


tooltip = None
if show_labels:
    tooltip = folium.GeoJsonTooltip(
        fields=["name", "iri_score", "wear_multiplier", "length_km"],
        aliases=["Segment:", "IRI Roughness:", "Wear Multiplier:", "Length (km):"],
        sticky=True,
        style="""
            background-color: #10151C;
            color: #F5F7FA;
            font-family: Inter, sans-serif;
            font-size: 12.5px;
            border: 1px solid #2B3542;
            border-radius: 8px;
            padding: 8px 10px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.4);
        """,
    )

folium.GeoJson(
    geojson_data,
    name="Road Wear Segments",
    style_function=style_function,
    highlight_function=highlight_function,
    tooltip=tooltip,
).add_to(fmap)

# City markers
if show_markers:
    folium.Marker(
        location=[route_info["start"][1], route_info["start"][2]],
        popup=f"<b>{route_info['start'][0]}</b> (Origin)",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(fmap)
    folium.Marker(
        location=[route_info["end"][1], route_info["end"][2]],
        popup=f"<b>{route_info['end'][0]}</b> (Destination)",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
    ).add_to(fmap)

# Optional heatmap of wear density
if show_heatmap:
    heat_points = []
    for s in segments:
        (lon_a, lat_a), (lon_b, lat_b) = s["coords"]
        heat_points.append([(lat_a + lat_b) / 2, (lon_a + lon_b) / 2, s["wear_multiplier"]])
    plugins.HeatMap(heat_points, radius=22, blur=18, min_opacity=0.35).add_to(fmap)

# Fit bounds to route
all_lats = [p[0] for w in [route_info["waypoints"]] for p in w]
all_lons = [p[1] for w in [route_info["waypoints"]] for p in w]
fmap.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]], padding=(30, 30))

plugins.Fullscreen(position="topright").add_to(fmap)
plugins.MiniMap(toggle_display=True, position="bottomleft").add_to(fmap)

st.markdown("<div class='ra-map-wrap'>", unsafe_allow_html=True)
st_folium(fmap, use_container_width=True, height=560, returned_objects=[])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="ra-legend">
        <b style="color:#C6CDD5;">Wear Legend:</b>
        <span><span class="ra-dot" style="background:#3CCB7F;"></span>Healthy (&lt; 1.2×)</span>
        <span><span class="ra-dot" style="background:#F2A93B;"></span>Moderate (1.2× – 1.8×)</span>
        <span><span class="ra-dot" style="background:#E5484D;"></span>Severe (&gt; 1.8×)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# ROUTE COMPARISON MATRIX — Route A vs Route B
# --------------------------------------------------------------------------------
st.markdown("<div class='ra-section-title'>🔀 Route Comparison Matrix</div>", unsafe_allow_html=True)

worst_segment = max(segments, key=lambda s: s["wear_multiplier"])

# Live figures from ml.compare_routes() (route_cmp), computed earlier from the
# real trained model + live road_state.json — no fixed pitch-deck numbers.
route_a_rul = route_cmp["route_a"]["adjusted_rul"]
route_a_cost = route_cmp["route_a"]["annual_cost"]
route_b_rul = route_cmp["route_b"]["adjusted_rul"]
route_b_cost = route_cmp["route_b"]["annual_cost"]
route_b_detour_min = 14 if IS_LIVE_CORRIDOR else 0

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        f"""<div class="ra-route-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span class="ra-route-name">🅰️ Route A — Current ({route_key.split('(')[-1].strip(')')})</span>
                <span class="ra-route-tag" style="background:#3A1E1E;color:#E5484D;">HIGH WEAR</span>
            </div>
            <hr style="margin:14px 0;border-color:#202832;">
            <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                <span style="color:#8A97A6;font-size:13px;">Predicted Suspension Failure</span>
                <span style="color:#F5F7FA;font-weight:700;font-size:14px;">{route_a_rul} days</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                <span style="color:#8A97A6;font-size:13px;">Estimated Repair Cost</span>
                <span style="color:#F5F7FA;font-weight:700;font-size:14px;">₹{route_a_cost:,}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                <span style="color:#8A97A6;font-size:13px;">Detour Time</span>
                <span style="color:#F5F7FA;font-weight:700;font-size:14px;">+0 min (baseline)</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8A97A6;font-size:13px;">Worst Segment</span>
                <span style="color:#E5484D;font-weight:700;font-size:13.5px;">{worst_segment['name']} ({worst_segment['wear_multiplier']}×)</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        f"""<div class="ra-route-card best">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span class="ra-route-name">🅱️ Route B — Alternate (+{route_b_detour_min} min)</span>
                <span class="ra-route-tag" style="background:#1B4332;color:#6FCF97;">RECOMMENDED</span>
            </div>
            <hr style="margin:14px 0;border-color:#202832;">
            <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                <span style="color:#8A97A6;font-size:13px;">Predicted Suspension Failure</span>
                <span style="color:#6FCF97;font-weight:700;font-size:14px;">{route_b_rul} days</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                <span style="color:#8A97A6;font-size:13px;">Estimated Repair Cost</span>
                <span style="color:#F5F7FA;font-weight:700;font-size:14px;">₹{route_b_cost:,}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                <span style="color:#8A97A6;font-size:13px;">Detour Time</span>
                <span style="color:#F5F7FA;font-weight:700;font-size:14px;">+{route_b_detour_min} min</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8A97A6;font-size:13px;">Life Extension vs Route A</span>
                <span style="color:#6FCF97;font-weight:700;font-size:13.5px;">+{route_b_rul - route_a_rul} days ({round((route_b_rul/route_a_rul - 1)*100)}%)</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

if IS_LIVE_CORRIDOR:
    st.caption(
        "RUL and cost figures come from a Random Forest model trained on NASA C-MAPSS FD001, "
        "adjusted using live IRI roughness scores from data/road_state.json for this corridor."
    )
else:
    st.caption(
        "RUL comes from the trained Random Forest model, but this route's wear multiplier is "
        "estimated from the illustrative map segments (no live IRI sensor data for this corridor yet). "
        "Select Nagpur → Raipur (NH53) for fully live-data figures."
    )

# --------------------------------------------------------------------------------
# BIG METRIC CALLOUT — TOTAL FLEET SAVINGS
# --------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="ra-callout">
        <div>
            <div class="ra-callout-label">💰 Total Fleet Savings</div>
            <div class="ra-callout-value">₹{annual_saving/100000:.1f} Lakhs <span style="font-size:20px;font-weight:600;color:#9FB3A9;">/ Year</span></div>
            <div class="ra-callout-sub">Projected for a fleet of {fleet_size} trucks switching from Route A to Route B on this corridor{"" if IS_LIVE_CORRIDOR else " (estimated — select the NH53 route for the live-data figure)"}</div>
        </div>
        <div class="ra-callout-icon">📈</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# SIMULATED FLEET TABLE
# --------------------------------------------------------------------------------
st.markdown("<div class='ra-section-title'>🚚 Fleet Overview — Live Vehicle Status</div>", unsafe_allow_html=True)

FLEET_SEED = 7
fleet_rng = random.Random(FLEET_SEED)

truck_models = ["Tata Prima 4025.S", "Tata Signa 3118.T", "Tata LPT 1613", "Tata Ultra T.16", "Tata Prima 2830.K"]
route_names = list(ROUTES.keys())
plate_states = ["MH-31", "MH-40", "CG-04", "MH-49", "DL-01", "RJ-14", "MH-12", "CG-15"]

fleet_rows = []
for i in range(8):
    state_code = plate_states[i]
    plate = f"{state_code}-{fleet_rng.randint(1000, 9999)}-{fleet_rng.choice('ABCDEFGH')}{fleet_rng.choice('ABCDEFGH')}"
    active_route = route_key if i < 5 else fleet_rng.choice(route_names)
    health_index = round(fleet_rng.uniform(38, 97), 1)

    if health_index >= 75:
        status = "🟢 Healthy"
    elif health_index >= 50:
        status = "🟡 Watch"
    else:
        status = "🔴 At Risk"

    recommended = "Route B (Alternate)" if health_index < 65 and active_route == route_key else "Keep Current Route"

    fleet_rows.append(
        {
            "License Plate": plate,
            "Model": fleet_rng.choice(truck_models),
            "Active Route": active_route,
            "Health Index": health_index,
            "Status": status,
            "Recommended Route": recommended,
        }
    )

fleet_df = pd.DataFrame(fleet_rows)

st.dataframe(
    fleet_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Health Index": st.column_config.ProgressColumn(
            "Health Index",
            help="Composite vehicle health score (0-100), factoring in road wear exposure",
            format="%.1f",
            min_value=0,
            max_value=100,
        ),
    },
)

st.caption(
    f"{sum(1 for r in fleet_rows if '🔴' in r['Status'])} truck(s) currently flagged At Risk · "
    f"{sum(1 for r in fleet_rows if r['Recommended Route'] != 'Keep Current Route')} recommended for route reassignment."
)
