# Tata-InnoVent-RoadAware

**Road-Aware Vehicle Maintenance AI**
*The first system that treats Indian roads as a variable in predicting vehicle wear.*

> Tata InnoVent Challenge 2026 | Edge AI for Vehicle Health

---

## 🎯 One-Line Pitch

We tell fleet owners not just *when* their vehicle parts will fail, but *which roads are killing their vehicles* and *which alternate route saves them lakhs every year*.

---

## 🚧 The Problem

Imagine two identical Tata trucks. Same model, same age, same driver style.

- **Truck 1** runs Mumbai expressways.
- **Truck 2** runs Nagpur–Raipur on NH53 potholes, speed breakers every 500m, unpaved patches.

After 6 months, Truck 2's suspension is destroyed. Truck 1 is fine. **Every existing maintenance app treats both trucks the same way.** That's the problem.

| Gap | Why it matters |
|---|---|
| **Every model ignores Indian roads** | Predictive maintenance AI is trained on clean Western highway data — invalid for Indian conditions (potholes, overloading, kaccha roads). |
| **Predictions don't explain WHY** | Tools say "replace suspension in 20 days" but never say *why* — e.g. a specific 47km stretch with an IRI roughness score of 8.2. |
| **No link between roads and cost** | Nobody connects bad-road-driven wear to actual maintenance spend. |
| **Route decisions ignore vehicle health** | Drivers optimize for distance/time, never for the wear a route causes. |

---

## 💡 Our Solution

We assign every road segment in India a **wear score** — a number representing how badly that stretch damages a vehicle and use it to power three things no existing tool does:

1. **Smarter Prediction** Failure predictions are adjusted using actual road roughness, not just age/mileage. Bad roads → shorter predicted part life.
2. **Road Blame Report** — Pinpoints exactly which road segments are driving wear (e.g. *"The 47km stretch near Wardha is responsible for 38% of your suspension wear this month."*).
3. **Smarter Route Advice** — Compares 2–3 possible routes and recommends the one that's cheapest once road-wear damage is factored in — not just fuel and time.

### Example Dashboard Output

```
Route A (Current — NH53 via Wardha): Suspension failure predicted in 34 days. Repair cost: ₹18,000.
Route B (Alternate — adds 14 minutes): Suspension failure predicted in 67 days. Repair cost: ₹18,000.

Recommendation: Take Route B.
For a fleet of 10 trucks doing this route daily → saves ~₹2.3 lakhs/year in early suspension replacements.
```

---

## ⚙️ How It Works — The Pipeline

| Piece | Description |
|---|---|
| **1. Road Intelligence Layer** | Downloads road data for a route (e.g. Nagpur–Raipur) from OpenStreetMap + PMGSY government road quality data. Assigns an **IRI score** (roughness/damage rating) to every road segment. |
| **2. Wear Rate Calculator** | Converts IRI scores into a **wear multiplier** using published IRI-to-wear-rate research. A badly potholed segment might carry a 2.5x wear multiplier vs. a smooth highway. Physics-based, not guesswork. |
| **3. Adjusted RUL Predictor** | Takes a standard RUL (Remaining Useful Life) model trained on NASA's C-MAPSS dataset and adjusts its output using the wear multipliers from Piece 2. Same vehicle, different routes → different predictions. |
| **4. Route Optimizer** | Generates 2–3 alternate routes via OpenRouteService, computes total wear cost + time cost per route, and recommends the best balance of travel time and vehicle health. |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core language |
| **OSMnx + GeoPandas** | Download/process road map data from OpenStreetMap |
| **PMGSY Open Data** | Indian government road quality ratings |
| **NASA C-MAPSS Dataset** | Public sensor-degradation dataset for RUL model training |
| **scikit-learn** | Train the RUL prediction model |
| **NumPy** | Build the IRI-to-wear-rate formula |
| **NetworkX** | Model roads as a graph; route optimization with wear-cost weights |
| **OpenRouteService API** | Generate alternate routes between cities |
| **Folium** | Interactive maps with wear-score colour overlays |
| **Streamlit** | Full web dashboard, no frontend code needed |

---

## 🗺️ Build Roadmap

| Step | Time | Task |
|---|---|---|
| 1 | 2–3 hrs | Download NASA C-MAPSS + OSMnx road graph (Nagpur–Raipur); set up Python env |
| 2 | 2–3 hrs | Build the road wear score system (IRI scores + wear multiplier formula) |
| 3 | 2–3 hrs | Train RUL model (Random Forest on C-MAPSS) and apply the wear multiplier adjustment |
| 4 | 1–2 hrs | Build route generator + optimizer using OpenRouteService |
| 5 | 2–3 hrs | Build Streamlit dashboard (wear map, route comparison, simulated fleet view) |
| 6 | 1–2 hrs | Polish demo: clickable "repair this road, save ₹X/year" interaction; rehearse pitch |

**Core formula:**
```
adjusted_RUL = base_RUL / road_wear_multiplier
```

---

## 🏆 What Makes This Different

- **Real Indian data, not generic datasets** — Uses PMGSY government road data on top of NASA C-MAPSS, making the model actually valid for Indian conditions.
- **Road as a variable, not a constant** — Every other predictive maintenance system ignores the road entirely. This project's core insight is treating road roughness as a primary input, not background noise.
- **Decisions, not just predictions** — Instead of stopping at "part fails in X days," this system recommends an actionable alternate route with a rupee savings figure attached.
- **Built for a real corridor** — Designed specifically around the Nagpur–Raipur corridor, a real, documented, high-wear Indian route — not a hypothetical fleet.

---

## 🌟 The Novelty

> Road roughness is not background noise in vehicle maintenance, it is the primary input.

Every existing system (including those used by Tata, Mahindra, and global fleet companies) models wear as a function of time, mileage, and load. In developed countries with good roads, this is fine. **In India, it isn't.** No one has built a system that models wear as a function of the actual road segments a vehicle travels until now.

---

## 🎤 2-Minute Pitch Structure

| Segment | Time | Content |
|---|---|---|
| **Hook** | 15s | Two identical trucks, one road destroys the suspension in 6 months, one doesn't — existing apps can't tell the difference. |
| **Problem** | 20s | Predictive maintenance AI is built for Western roads and gives wrong predictions in India. |
| **Solution** | 30s | Wear score per road segment → adjusted maintenance predictions → smarter route recommendations. |
| **Demo** | 40s | Show the map, Route A vs Route B, the rupee savings number, and a live click-to-simulate road repair interaction. |
| **Close** | 15s | First system built specifically for Indian roads and fleets — 100% software, deployable today. |

---

## 📌 Status

🚧 Hackathon project — built for the **Tata InnoVent Challenge 2026**.

---

## 📄 License

TBD
