# Tata-InnoVent-RoadAware

**Road-Aware Vehicle Maintenance AI**

An edge AI system that models Indian road quality as a direct input to vehicle maintenance prediction, submitted for the Tata InnoVent Challenge 2026.

## Overview

Predictive maintenance systems typically estimate component wear using vehicle age, mileage, and load, with models trained on Western road data. This approach does not generalize to Indian conditions, where road roughness varies significantly and materially affects wear rates. This project treats road quality as a primary variable rather than a constant, enabling route-specific maintenance predictions and cost-aware routing recommendations.

## Problem Statement

- Existing predictive maintenance models are trained on Western highway data and do not account for Indian road conditions.
- Failure predictions typically lack a causal explanation, making root-cause diagnosis difficult.
- No existing tool connects road quality data to maintenance cost.
- Route planning is optimized for time and distance, not vehicle wear.

## Solution

The system assigns a wear score to individual road segments and uses it to:

1. **Adjust failure predictions** based on the roughness of roads a vehicle has traveled.
2. **Attribute wear to specific road segments**, identifying which stretches contribute most to component degradation.
3. **Recommend routes** that minimize total cost when accounting for wear-related maintenance expense, not only time and fuel.

### Sample Output

```
Route A (NH53 via Wardha): Suspension failure predicted in 34 days. Estimated repair cost: ₹18,000.
Route B (Alternate, +14 min): Suspension failure predicted in 67 days. Estimated repair cost: ₹18,000.

Recommendation: Route B. For a fleet of 10 trucks on this route daily,
estimated annual savings: ₹2.3 lakhs.
```

## System Architecture

| Component | Function |
|---|---|
| Road Intelligence Layer | Retrieves road geometry from OpenStreetMap and road quality data from PMGSY; assigns an IRI (International Roughness Index) score to each segment. |
| Wear Rate Calculator | Converts IRI scores into a wear multiplier using published IRI–wear-rate correlations. |
| Adjusted RUL Predictor | Adjusts a Remaining Useful Life model (trained on the NASA C-MAPSS dataset) using segment-level wear multipliers. |
| Route Optimizer | Generates alternate routes via OpenRouteService and ranks them by combined wear cost and time cost. |

## Technology Stack

| Tool | Purpose |
|---|---|
| Python | Core implementation language |
| OSMnx, GeoPandas | Road network data acquisition and processing |
| PMGSY Open Data | Indian government road quality data |
| NASA C-MAPSS Dataset | Training data for the RUL model |
| scikit-learn | RUL model training |
| NumPy | Wear-rate formula implementation |
| NetworkX | Graph-based route modeling |
| OpenRouteService API | Alternate route generation |
| Folium | Map visualization |
| Streamlit | Dashboard interface |
