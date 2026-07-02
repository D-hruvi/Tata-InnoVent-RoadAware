"""
Road-Aware Vehicle Maintenance AI — Dashboard
Tata InnoVent Challenge 2026

A Streamlit + Folium dashboard that visualizes Indian road segments coloured
by their "wear multiplier" (how badly a stretch of road accelerates vehicle
component wear), and lets a fleet owner compare routes.
"""

import random
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium

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

    st.markdown("<div class='ra-section-title'>🗺️ Map Layers</div>", unsafe_allow_html=True)
    show_markers = st.checkbox("Show city markers", value=True)
    show_heatmap = st.checkbox("Show wear density heatmap", value=False)
    show_labels = st.checkbox("Show segment tooltips", value=True)

    st.markdown("<div class='ra-section-title'>⚙️ Model</div>", unsafe_allow_html=True)
    st.caption("Base RUL model: Random Forest — NASA C-MAPSS")
    st.caption("Road data: OSM + PMGSY roughness index")

    st.markdown("---")
    if st.button("🔄 Refresh Wear Scores", use_container_width=True):
        st.session_state["seed"] = random.randint(0, 10_000)

seed = st.session_state.get("seed", 42)
segments = build_segments(route_key, seed=seed)
geojson_data = segments_to_geojson(segments)
route_info = ROUTES[route_key]

# --------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------
avg_wear = sum(s["wear_multiplier"] for s in segments) / len(segments)
severe_count = sum(1 for s in segments if s["wear_multiplier"] > 1.8)
total_km = sum(s["length_km"] for s in segments)
base_rul_days = 90
adjusted_rul_days = max(int(base_rul_days / avg_wear), 1)
repair_cost = 18000
annual_saving = int(fleet_size * daily_trips * 52 * (repair_cost / max(adjusted_rul_days, 1)) * 0.28)

st.markdown(
    f"""
    <div class="ra-header">
        <div>
            <p class="ra-title">Road-Aware Vehicle Maintenance AI</p>
            <p class="ra-subtitle">Live wear intelligence for <b style="color:#C6CDD5;">{route_info['start'][0]} → {route_info['end'][0]}</b> &nbsp;·&nbsp; {len(segments)} segments monitored</p>
        </div>
        <div class="ra-badge">● LIVE MODEL</div>
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
            <div class="ra-card-label">Adjusted RUL</div>
            <div class="ra-card-value">{adjusted_rul_days} days</div>
            <div class="ra-card-delta-bad">↓ {base_rul_days - adjusted_rul_days} days vs. baseline</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class="ra-card">
            <div class="ra-card-label">Avg Wear Multiplier</div>
            <div class="ra-card-value">{avg_wear:.2f}×</div>
            <div class="ra-card-sub">across {len(segments)} segments · {total_km:.0f} km</div>
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
# ROUTE COMPARISON
# --------------------------------------------------------------------------------
st.markdown("<div class='ra-section-title'>🔀 Route Comparison</div>", unsafe_allow_html=True)

worst_segment = max(segments, key=lambda s: s["wear_multiplier"])
route_a_rul = adjusted_rul_days
route_b_rul = int(adjusted_rul_days * 1.9)
route_a_cost = repair_cost
route_b_cost = repair_cost

rc1, rc2 = st.columns(2)
with rc1:
    st.markdown(
        f"""<div class="ra-route-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span class="ra-route-name">Route A — Current ({route_key.split('(')[-1].strip(')')})</span>
                <span class="ra-route-tag" style="background:#3A1E1E;color:#E5484D;">HIGH WEAR</span>
            </div>
            <div style="margin-top:12px;color:#C6CDD5;font-size:13.5px;">
                Predicted suspension failure in <b>{route_a_rul} days</b><br>
                Estimated repair cost: <b>₹{route_a_cost:,}</b><br>
                Worst segment: <b>{worst_segment['name']}</b> ({worst_segment['wear_multiplier']}× wear)
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
with rc2:
    st.markdown(
        f"""<div class="ra-route-card best">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span class="ra-route-name">Route B — Alternate (+14 min)</span>
                <span class="ra-route-tag" style="background:#1B4332;color:#6FCF97;">RECOMMENDED</span>
            </div>
            <div style="margin-top:12px;color:#C6CDD5;font-size:13.5px;">
                Predicted suspension failure in <b>{route_b_rul} days</b><br>
                Estimated repair cost: <b>₹{route_b_cost:,}</b><br>
                Fleet savings: <b style="color:#6FCF97;">₹{annual_saving:,} / year</b> for {fleet_size} trucks
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

st.caption(
    "All figures are model-generated estimates for demo purposes, based on mock IRI roughness scores "
    "and a wear-adjusted Remaining Useful Life (RUL) calculation."
)
