# =============================================================================
# ml_engine.py
# PERSON 2 — ML Model for Road-Aware Vehicle Maintenance AI
# Tata InnoVent Challenge 2026
# =============================================================================
# What this file does:
#   1. Loads NASA C-MAPSS FD001 dataset
#   2. Engineers features (rolling stats on sensor readings)
#   3. Trains a Random Forest model to predict RUL (Remaining Useful Life)
#   4. Adjusts RUL based on road roughness (IRI score) — THE CORE INNOVATION
#   5. Manages live road condition updates — roads can be repaired/worsened
#   6. Simulates repair impact — how much money does fixing a road save?
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import joblib
import os
import json

# ── Column names for C-MAPSS dataset ─────────────────────────────────────────
# FD001 has no header row so we define column names manually
COLS = (
    ["engine_id", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

# These 14 sensors actually change as the engine degrades
# The other 7 are constant and useless for prediction
USEFUL_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"
]

# ── Road state file location ──────────────────────────────────────────────────
ROAD_STATE_FILE = "data/road_state.json"

# ── Default road IRI scores for NH53 Nagpur→Raipur corridor ──────────────────
# IRI = International Roughness Index
# Scale: 0-2 = Very Good, 2-4 = Good, 4-6 = Moderate, 6-8 = Fair, 8+ = Poor
DEFAULT_ROAD_STATE = {
    "NH53_Nagpur_Wardha": {
        "iri_score": 8.5,
        "condition": "Poor",
        "last_updated": "2026-07-01",
        "repaired": False,
        "notes": "Known pothole stretch near Wardha — major damage"
    },
    "NH53_Wardha_Chandrapur": {
        "iri_score": 6.2,
        "condition": "Fair",
        "last_updated": "2026-07-01",
        "repaired": False,
        "notes": "Moderate wear, speed breakers every 2km"
    },
    "NH53_Chandrapur_Raipur": {
        "iri_score": 4.1,
        "condition": "Moderate",
        "last_updated": "2026-07-01",
        "repaired": False,
        "notes": "Better surface, some rough patches near state border"
    },
    "NH353_Alternate_Full": {
        "iri_score": 3.8,
        "condition": "Good",
        "last_updated": "2026-07-01",
        "repaired": False,
        "notes": "Alternate route via NH353 — smoother, adds ~14 minutes"
    },
}


# =============================================================================
# SECTION 1 — DATA LOADING & LABELING
# =============================================================================

def load_data(train_path="data/train_FD001.txt",
              rul_path="data/RUL_FD001.txt"):
    """
    Loads NASA C-MAPSS FD001 dataset and computes RUL for every cycle.

    Logic:
        - Each engine runs until it fails
        - Max cycle for engine X = when it failed
        - RUL at any cycle = max_cycle - current_cycle
        - Cap RUL at 125 (engines are 'healthy' before that — standard practice)
    """
    train = pd.read_csv(
        train_path, sep=r"\s+", header=None, names=COLS
    )

    # Find failure cycle for each engine
    max_cycles = (
        train.groupby("engine_id")["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "max_cycle"})
    )
    train = train.merge(max_cycles, on="engine_id")

    # RUL = how many cycles left before failure
    train["RUL"] = train["max_cycle"] - train["cycle"]

    # Cap at 125 — standard C-MAPSS preprocessing
    train["RUL"] = train["RUL"].clip(upper=125)

    return train


# =============================================================================
# SECTION 2 — FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """
    Adds rolling mean and std features for each useful sensor.

    Why rolling stats?
        A single sensor reading doesn't tell you much.
        But a sensor that's been TRENDING downward over the last 3 cycles
        tells you the engine is degrading. That's what rolling mean/std captures.
    """
    df = df.copy()
    for sensor in USEFUL_SENSORS:
        # Rolling mean — captures trend direction
        df[f"{sensor}_mean3"] = (
            df.groupby("engine_id")[sensor]
            .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )
        # Rolling std — captures how erratic the readings are becoming
        df[f"{sensor}_std3"] = (
            df.groupby("engine_id")[sensor]
            .transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
        )
    return df


# =============================================================================
# SECTION 3 — TRAIN & SAVE MODEL
# =============================================================================

def train_and_save(train_path="data/train_FD001.txt",
                   rul_path="data/RUL_FD001.txt",
                   model_path="models/rul_model.pkl"):
    """
    Trains a Random Forest Regressor on C-MAPSS data and saves it.
    Call this ONCE — it creates rul_model.pkl in /models/

    Saves model + scaler + feature list together so they always stay in sync.
    """
    print("Loading data...")
    df = load_data(train_path, rul_path)

    print("Engineering features...")
    df = engineer_features(df)

    # Build feature column list
    feature_cols = (
        USEFUL_SENSORS
        + [f"{s}_mean3" for s in USEFUL_SENSORS]
        + [f"{s}_std3"  for s in USEFUL_SENSORS]
        + ["cycle"]
    )

    X = df[feature_cols]
    y = df["RUL"]

    # Scale to 0-1 range — helps Random Forest work better
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1          # use all CPU cores — faster training
    )
    model.fit(X_scaled, y)

    # Quick evaluation
    preds = model.predict(X_scaled)
    mae = mean_absolute_error(y, preds)
    print(f"✅ Training MAE: {mae:.2f} cycles")

    # Save model + scaler + feature list as one bundle
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model":        model,
        "scaler":       scaler,
        "feature_cols": feature_cols
    }, model_path)
    print(f"✅ Model saved to {model_path}")
    return mae


# =============================================================================
# SECTION 4 — ROAD CONDITION MANAGER
# =============================================================================

def load_road_state() -> dict:
    """
    Loads current IRI scores from file.
    If file doesn't exist yet, creates it with default values.

    This is how the model 'remembers' road conditions between sessions.
    When a road gets repaired, we update this file — predictions update instantly.
    No retraining needed.
    """
    if os.path.exists(ROAD_STATE_FILE):
        with open(ROAD_STATE_FILE, "r") as f:
            return json.load(f)
    else:
        save_road_state(DEFAULT_ROAD_STATE)
        return DEFAULT_ROAD_STATE


def save_road_state(state: dict):
    """Saves road condition state to JSON file."""
    os.makedirs("data", exist_ok=True)
    with open(ROAD_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def update_road_condition(segment_name: str,
                          new_iri_score: float,
                          reason: str = "Manual update") -> dict:
    """
    Permanently updates IRI score for a road segment.

    Use this when:
        - A road actually gets repaired (IRI drops)
        - Monsoon season damages a road (IRI rises)
        - Government publishes new road quality survey data

    The beauty: no ML retraining needed.
    Just update the IRI → predictions automatically improve/worsen.

    Example:
        update_road_condition("NH53_Nagpur_Wardha", 3.0, "NHAI resurfacing done")
    """
    state = load_road_state()

    if segment_name not in state:
        return {"error": f"Segment '{segment_name}' not found in road state"}

    old_iri = state[segment_name]["iri_score"]

    # Update values
    state[segment_name]["iri_score"]    = new_iri_score
    state[segment_name]["last_updated"] = "2026-07-02"
    state[segment_name]["repaired"]     = new_iri_score < old_iri
    state[segment_name]["notes"]        = reason

    # Auto-assign condition label based on IRI
    if new_iri_score <= 3:
        state[segment_name]["condition"] = "Good"
    elif new_iri_score <= 5:
        state[segment_name]["condition"] = "Moderate"
    elif new_iri_score <= 7:
        state[segment_name]["condition"] = "Fair"
    else:
        state[segment_name]["condition"] = "Poor"

    save_road_state(state)

    print(f"✅ Road updated: {segment_name}")
    print(f"   IRI: {old_iri} → {new_iri_score}")
    print(f"   Condition: {state[segment_name]['condition']}")
    print(f"   Reason: {reason}")

    return state[segment_name]


def get_all_segments() -> dict:
    """Returns all road segments with current IRI scores. Used by dashboard."""
    return load_road_state()


def get_route_iri(route: str = "A") -> tuple:
    """
    Gets LIVE average IRI and wear multiplier for a route.

    Route A = NH53 direct (3 segments averaged)
    Route B = NH353 alternate (1 segment)

    Returns: (avg_iri, wear_multiplier)

    This replaces hardcoded multipliers — uses live road state file instead.
    So when a road gets updated, this automatically reflects the new value.
    """
    state = load_road_state()

    if route == "A":
        segments = [
            "NH53_Nagpur_Wardha",
            "NH53_Wardha_Chandrapur",
            "NH53_Chandrapur_Raipur"
        ]
        avg_iri = sum(state[s]["iri_score"] for s in segments) / len(segments)
    else:
        avg_iri = state["NH353_Alternate_Full"]["iri_score"]

    # IRI → Wear Multiplier formula
    # Based on published IRI-vehicle wear rate correlations
    # Smooth road (IRI=2) → multiplier 1.24 (24% extra wear vs ideal)
    # Bad road (IRI=9)   → multiplier 2.08 (2x extra wear)
    wear_multiplier = round(1 + (0.12 * avg_iri), 3)

    return round(avg_iri, 2), wear_multiplier


# =============================================================================
# SECTION 5 — RUL PREDICTION (THE CORE INNOVATION)
# =============================================================================

def calculate_adjusted_rul(vehicle_age_days: int,
                            load_pct: float,
                            wear_multiplier: float,
                            model_path: str = "models/rul_model.pkl") -> dict:
    """
    THE KEY FUNCTION — imported by Person 3 into the dashboard.

    Takes:
        vehicle_age_days  — how old the vehicle is in days
        load_pct          — how loaded (50% = half load, 120% = overloaded)
        wear_multiplier   — from get_route_iri() — higher = rougher road

    Returns dict with:
        base_rul          — prediction ignoring road quality
        adjusted_rul      — prediction WITH road quality (THE INNOVATION)
        component_health  — 0 to 100% health score
        risk_level        — LOW / MEDIUM / HIGH / CRITICAL
        recommendation    — plain English action for the fleet manager

    The core formula:
        adjusted_rul = base_rul / wear_multiplier

        If wear_multiplier = 1.86 (rough road):
            adjusted_rul = base_rul / 1.86  →  shorter life
        If wear_multiplier = 1.46 (smoother road):
            adjusted_rul = base_rul / 1.46  →  longer life

        Same vehicle, different road = different prediction.
        That difference IS the proof of concept.
    """
    # Load trained model bundle
    bundle       = joblib.load(model_path)
    model        = bundle["model"]
    scaler       = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    # Convert vehicle age to cycles (1 cycle ≈ 1 hr, ~8 hrs/day)
    cycles      = vehicle_age_days * 8
    degradation = min(cycles / 1000, 1.0)   # 0→1 as vehicle ages
    load_factor = load_pct / 100.0

    # Simulate realistic sensor readings based on age + load
    # These values mimic how C-MAPSS sensors degrade over time
    sensor_vals = {
        "s2":  390  - 20  * degradation * load_factor,
        "s3":  1000 - 50  * degradation,
        "s4":  1400 - 100 * degradation * load_factor,
        "s7":  554  - 10  * degradation,
        "s8":  2388 - 30  * degradation,
        "s9":  9065 - 200 * degradation * load_factor,
        "s11": 47   - 5   * degradation,
        "s12": 522  - 20  * degradation * load_factor,
        "s13": 2388 - 30  * degradation,
        "s14": 8138 - 100 * degradation * load_factor,
        "s15": 8.4  + 2   * degradation,
        "s17": 392  - 15  * degradation,
        "s20": 39   - 5   * degradation * load_factor,
        "s21": 23   - 3   * degradation,
    }

    # Build feature row
    row = {}
    for s in USEFUL_SENSORS:
        row[s]              = sensor_vals[s]
        row[f"{s}_mean3"]   = sensor_vals[s] * (1 - 0.02 * degradation)
        row[f"{s}_std3"]    = abs(sensor_vals[s] * 0.01 * degradation)
    row["cycle"] = cycles

    X        = pd.DataFrame([row])[feature_cols]
    X_scaled = scaler.transform(X)

    # Base prediction (road-agnostic)
    base_rul = float(model.predict(X_scaled)[0])
    base_rul = max(base_rul, 1)

    # ── THE CORE INNOVATION ───────────────────────────────────────────────────
    # Adjust for road roughness
    adjusted_rul = base_rul / wear_multiplier
    adjusted_rul = max(round(adjusted_rul, 1), 1)
    base_rul     = round(base_rul, 1)
    # ─────────────────────────────────────────────────────────────────────────

    # Health score (0-100%)
    component_health = min(100, round((adjusted_rul / 125) * 100, 1))

    # Risk classification
    if adjusted_rul > 60:
        risk = "LOW"
        rec  = "No immediate action needed. Schedule next check in 30 days."
    elif adjusted_rul > 30:
        risk = "MEDIUM"
        rec  = "Plan maintenance within 2 weeks. Monitor closely."
    elif adjusted_rul > 10:
        risk = "HIGH"
        rec  = "Schedule maintenance this week. Avoid long routes."
    else:
        risk = "CRITICAL"
        rec  = "Do not operate. Immediate inspection required."

    return {
        "base_rul":         base_rul,
        "adjusted_rul":     adjusted_rul,
        "component_health": component_health,
        "risk_level":       risk,
        "recommendation":   rec,
        "wear_multiplier":  wear_multiplier,
    }


# =============================================================================
# SECTION 6 — ROUTE COMPARISON & SAVINGS CALCULATOR
# =============================================================================

def compare_routes(route_a_multiplier: float,
                   route_b_multiplier: float,
                   vehicle_age_days:   int,
                   load_pct:           float) -> dict:
    """
    Compares Route A vs Route B and calculates annual rupee savings.
    Powers Page 4 of the Streamlit dashboard.

    Cost assumptions (realistic Indian fleet context):
        ₹18,000 per suspension repair
        300 trips per year per truck
    """
    result_a = calculate_adjusted_rul(vehicle_age_days, load_pct, route_a_multiplier)
    result_b = calculate_adjusted_rul(vehicle_age_days, load_pct, route_b_multiplier)

    rul_a = result_a["adjusted_rul"]
    rul_b = result_b["adjusted_rul"]

    days_saved = round(rul_b - rul_a, 1)

    # Annual cost calculation
    cost_per_repair = 18000     # ₹ per suspension repair
    trips_per_year  = 300       # typical commercial fleet truck

    failures_a    = trips_per_year / max(rul_a, 1)
    failures_b    = trips_per_year / max(rul_b, 1)
    annual_cost_a = round(failures_a * cost_per_repair)
    annual_cost_b = round(failures_b * cost_per_repair)
    rupees_saved  = max(annual_cost_a - annual_cost_b, 0)

    better_route = "B" if rul_b > rul_a else "A"

    return {
        "route_a": {
            "wear_multiplier": route_a_multiplier,
            "adjusted_rul":    rul_a,
            "risk_level":      result_a["risk_level"],
            "annual_cost":     annual_cost_a,
        },
        "route_b": {
            "wear_multiplier": route_b_multiplier,
            "adjusted_rul":    rul_b,
            "risk_level":      result_b["risk_level"],
            "annual_cost":     annual_cost_b,
        },
        "days_saved":             days_saved,
        "rupees_saved_annually":  rupees_saved,
        "rupees_saved_fleet10":   rupees_saved * 10,
        "recommended_route":      better_route,
    }


# =============================================================================
# SECTION 7 — REPAIR IMPACT SIMULATOR
# =============================================================================

def simulate_repair_impact(segment_name:     str,
                            repaired_iri:     float,
                            vehicle_age_days: int   = 365,
                            load_pct:         float = 80,
                            fleet_size:       int   = 10) -> dict:
    """
    THE WOW FEATURE — powers Page 5 (Road Repair Simulator) in the dashboard.

    Answers: 'If this road segment gets repaired and IRI drops to X,
              how much money does the fleet save per year?'

    Does NOT permanently change road state — just simulates the scenario.
    The original road state is restored after calculation.

    This makes your model future-proof:
        - Road gets repaired → update IRI → predictions improve automatically
        - Monsoon damages road → update IRI → predictions worsen automatically
        - No retraining ever needed
    """
    state = load_road_state()

    if segment_name not in state:
        return {"error": f"Segment '{segment_name}' not found"}

    # ── Current situation ─────────────────────────────────────────────────────
    current_iri             = state[segment_name]["iri_score"]
    current_avg_iri_A, current_mult_A = get_route_iri("A")
    _, mult_B               = get_route_iri("B")
    current_comparison      = compare_routes(
        current_mult_A, mult_B, vehicle_age_days, load_pct
    )

    # ── Simulated situation ───────────────────────────────────────────────────
    # Deep copy so we don't corrupt the original
    state_copy = json.loads(json.dumps(state))
    state_copy[segment_name]["iri_score"] = repaired_iri
    save_road_state(state_copy)

    repaired_avg_iri_A, repaired_mult_A = get_route_iri("A")
    repaired_comparison = compare_routes(
        repaired_mult_A, mult_B, vehicle_age_days, load_pct
    )

    # ── Restore original state ────────────────────────────────────────────────
    save_road_state(state)

    # ── Calculate impact ──────────────────────────────────────────────────────
    rul_improvement = round(
        repaired_comparison["route_a"]["adjusted_rul"] -
        current_comparison["route_a"]["adjusted_rul"], 1
    )

    cost_before   = current_comparison["route_a"]["annual_cost"]
    cost_after    = repaired_comparison["route_a"]["annual_cost"]
    cost_saved    = max(cost_before - cost_after, 0)

    # Direction label
    if repaired_iri < current_iri:
        direction = "repaired/improved"
    else:
        direction = "damaged/worsened"

    return {
        "segment":          segment_name,
        "current_iri":      current_iri,
        "repaired_iri":     repaired_iri,
        "iri_change":       round(current_iri - repaired_iri, 1),
        "direction":        direction,

        "before": {
            "avg_iri_route_a":  current_avg_iri_A,
            "route_a_rul":      current_comparison["route_a"]["adjusted_rul"],
            "route_a_cost":     cost_before,
        },

        "after": {
            "avg_iri_route_a":  repaired_avg_iri_A,
            "route_a_rul":      repaired_comparison["route_a"]["adjusted_rul"],
            "route_a_cost":     cost_after,
        },

        "rul_gain_days":        rul_improvement,
        "cost_saved_per_truck": cost_saved,
        "extra_savings_fleet":  cost_saved * fleet_size,

        "insight": (
            f"{'Repairing' if repaired_iri < current_iri else 'Damage on'} "
            f"{segment_name} (IRI {current_iri} → {repaired_iri}) "
            f"{'extends' if rul_improvement >= 0 else 'reduces'} vehicle life "
            f"by {abs(rul_improvement)} days. "
            f"Annual maintenance cost per truck: "
            f"₹{cost_before:,} → ₹{cost_after:,}. "
            f"Fleet of {fleet_size} trucks "
            f"{'saves' if cost_saved > 0 else 'loses'} "
            f"₹{cost_saved * fleet_size:,}/year."
        )
    }


# =============================================================================
# SECTION 8 — TEST / POC PROOF
# Run: python components/ml_engine.py
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  TATA INNOVENT 2026 — ML ENGINE POC TEST")
    print("  Road-Aware Vehicle Maintenance AI")
    print("=" * 60)

    # ── STEP 1: Train model ───────────────────────────────────────────────────
    print("\nSTEP 1: Training RUL model on NASA C-MAPSS FD001...")
    train_and_save()

    # ── STEP 2: Show live road state ──────────────────────────────────────────
    print("\nSTEP 2: Current road conditions (live from road_state.json)...")
    state = load_road_state()
    for seg, data in state.items():
        print(f"  {seg}")
        print(f"    IRI: {data['iri_score']}  |  Condition: {data['condition']}")

    # ── STEP 3: Get live route multipliers ────────────────────────────────────
    print("\nSTEP 3: Live route wear multipliers...")
    iri_A, mult_A = get_route_iri("A")
    iri_B, mult_B = get_route_iri("B")
    print(f"  Route A (NH53 direct)   — avg IRI: {iri_A} → multiplier: {mult_A}")
    print(f"  Route B (NH353 alternate) — avg IRI: {iri_B} → multiplier: {mult_B}")

    # ── STEP 4: POC PROOF ─────────────────────────────────────────────────────
    print("\nSTEP 4: POC PROOF — same vehicle, different road = different RUL...")
    comparison = compare_routes(mult_A, mult_B, 365, 80)
    print(f"\n  Route A RUL : {comparison['route_a']['adjusted_rul']} days")
    print(f"  Route B RUL : {comparison['route_b']['adjusted_rul']} days")
    print(f"  Days saved  : +{comparison['days_saved']} days by taking Route B")
    print(f"  Annual cost (Route A): ₹{comparison['route_a']['annual_cost']:,}")
    print(f"  Annual cost (Route B): ₹{comparison['route_b']['annual_cost']:,}")
    print(f"  Annual savings       : ₹{comparison['rupees_saved_annually']:,}")
    print(f"  Fleet of 10 trucks   : ₹{comparison['rupees_saved_fleet10']:,}/year")
    print(f"\n  ✅ Recommended route: Route {comparison['recommended_route']}")

    # ── STEP 5: Repair simulator ──────────────────────────────────────────────
    print("\nSTEP 5: Road repair simulation — what if NH53 near Wardha gets fixed?")
    repair_result = simulate_repair_impact(
        segment_name="NH53_Nagpur_Wardha",
        repaired_iri=3.0,
        fleet_size=10
    )
    print(f"\n  {repair_result['insight']}")
    print(f"  RUL gain per vehicle   : +{repair_result['rul_gain_days']} days")
    print(f"  Cost saved per truck   : ₹{repair_result['cost_saved_per_truck']:,}/year")
    print(f"  Total fleet savings    : ₹{repair_result['extra_savings_fleet']:,}/year")

    # ── STEP 6: Monsoon damage simulation ────────────────────────────────────
    print("\nSTEP 6: Monsoon damage simulation — road gets worse...")
    monsoon_result = simulate_repair_impact(
        segment_name="NH53_Wardha_Chandrapur",
        repaired_iri=9.5,       # worsened after monsoon
        fleet_size=10
    )
    print(f"\n  {monsoon_result['insight']}")

    # ── STEP 7: Permanently update road after real repair ─────────────────────
    print("\nSTEP 7: Permanently updating road state after actual NHAI repair...")
    update_road_condition(
        "NH53_Nagpur_Wardha",
        new_iri_score=3.0,
        reason="NHAI completed resurfacing — July 2026"
    )

    # Show updated predictions
    iri_A_new, mult_A_new = get_route_iri("A")
    comp_new = compare_routes(mult_A_new, mult_B, 365, 80)
    print(f"\n  After repair:")
    print(f"  Route A new RUL    : {comp_new['route_a']['adjusted_rul']} days")
    print(f"  New annual savings : ₹{comp_new['rupees_saved_annually']:,}")
    print(f"\n  ✅ Model updated automatically — zero retraining needed!")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  POC COMPLETE ")
    print("=" * 60)
    print(f"  Route A RUL    : {comparison['route_a']['adjusted_rul']} days")
    print(f"  Route B RUL    : {comparison['route_b']['adjusted_rul']} days")
    print(f"  Annual savings : ₹{comparison['rupees_saved_annually']:,} per truck")
    print(f"  Fleet savings  : ₹{comparison['rupees_saved_fleet10']:,} for 10 trucks")
  
    print("=" * 60)
    