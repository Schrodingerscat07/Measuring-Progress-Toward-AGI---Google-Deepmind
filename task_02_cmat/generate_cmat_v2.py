#!/usr/bin/env python3
"""
CMAT Dataset Generator v2.0 -- Cross-Modal Attention Triage (FINAL)
====================================================================
500 samples across 10 creative domains x 5 integration depths x 5 conflict levels.
Each sample: PNG image + text passage + question + deterministic answer.
The answer REQUIRES integrating information from BOTH modalities.

Domains:
  1. Space Mission Control    -- sensor telemetry + protocol overrides
  2. Alchemy Lab              -- reagent properties + recipe rules
  3. City Planning Board      -- district statistics + regulation memos
  4. Sports Analytics         -- player metrics + coach adjustments
  5. Archaeological Survey    -- excavation findings + field note corrections
  6. Deep Ocean Submersible   -- dive sensor data + emergency protocols
  7. Satellite Network        -- orbital telemetry + comms priority rules
  8. Air Traffic Control      -- flight metrics + routing directives
  9. Cybersecurity SOC        -- threat indicators + incident response
 10. Vineyard Harvest         -- grape metrics + winemaking rules

Usage:  python generate_cmat_v2.py
Output: task_02_cmat/images/*.png + task_02_cmat/metadata.jsonl
"""

import os, json, random, textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

random.seed(2026)
np.random.seed(2026)

SCRIPT_DIR = Path(__file__).parent
IMAGES_DIR = SCRIPT_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# DOMAIN CONFIGURATIONS (10 domains)
# =====================================================================

DOMAINS = {
    "space_mission": {
        "title": "DEEP SPACE VESSEL 'AURORA' -- SENSOR ARRAY",
        "bg": "#0a0e27", "fg": "#c8d6e5", "accent": "#00ff88",
        "header_bg": "#162044", "cell_bg": "#0f1535",
        "status_colors": {"good": "#00ff88", "warn": "#ffaa00", "bad": "#ff4444"},
        "entities": ["Thermal Array", "Pressure Hull", "O2 Recycler",
                      "Fuel Cell", "Radiation Shield", "Nav Beacon"],
        "attrs": ["Reading", "Baseline", "Drift (%)"],
        "ranges": [(-15, 75), (85, 125), (-8, 12)],
        "memo_from": "Chief Engineer Vasquez",
        "memo_re": "Sensor Calibration Override -- Cycle 47",
        "filler": "All crew: routine maintenance for Deck 7 rescheduled to 0300 UTC. "
                  "Hydroponics reports nominal yield. External hull EVA inspection completed. "
                  "Mess hall recycler online. Port-of-call ETA 14.2 standard days. "
                  "The new navigation firmware update passed QA on all three redundant cores. "
                  "Cargo manifest for Relay Station Kepler has been finalized and transmitted.",
    },
    "alchemy_lab": {
        "title": "GRAND ALCHEMIST'S WORKBENCH -- REAGENT LOG",
        "bg": "#1a1207", "fg": "#e8d5a3", "accent": "#ff9900",
        "header_bg": "#2d2010", "cell_bg": "#1f1809",
        "status_colors": {"good": "#ffcc00", "warn": "#ff8800", "bad": "#ff3300"},
        "entities": ["Dragon Petal", "Moon Salt", "Wyrm Venom",
                      "Starlight Dew", "Iron Bloom", "Ghost Moss"],
        "attrs": ["Volume (mL)", "Purity (%)", "Potency"],
        "ranges": [(10, 195), (55, 98), (1, 10)],
        "memo_from": "Archmage Elara",
        "memo_re": "Recipe Corrections -- Elixir of Clarity Batch 12",
        "filler": "Apprentice council extended Wednesday brewing hours. "
                  "Cauldron #4 east wing needs re-seasoning. Crystal vials arrive next Moonday. "
                  "Reagent spills must be reported to the Safety Warlock. "
                  "Annual familiar vaccination drive begins next tenday. "
                  "The enchanted ventilation system in Lab 3 has been repaired by the Rune Division.",
    },
    "city_planning": {
        "title": "METRO COUNCIL -- DISTRICT STATISTICS Q2 2026",
        "bg": "#f5f0e8", "fg": "#2c3e50", "accent": "#2980b9",
        "header_bg": "#d5cfc3", "cell_bg": "#ece7db",
        "status_colors": {"good": "#27ae60", "warn": "#f39c12", "bad": "#e74c3c"},
        "entities": ["Riverside Ward", "Northgate Hub", "Old Quarter",
                      "Tech Corridor", "Harbor District", "Greenfield Zone"],
        "attrs": ["Pop. (K)", "Buildings", "Green Cover (%)"],
        "ranges": [(8, 115), (30, 480), (5, 42)],
        "memo_from": "City Planner Dr. Okonkwo",
        "memo_re": "Quarterly Zoning Corrections & Projections",
        "filler": "Waterfront promenade hearing received 312 comments. "
                  "Bus route 14B extends to Greenfield in June. "
                  "Heritage applications for Old Quarter under review. "
                  "Cycling infrastructure proposal passed 5-2. "
                  "Next council meeting April 28th 7 PM City Hall Room B.",
    },
    "sports_analytics": {
        "title": "THUNDER FC -- PRE-MATCH PERFORMANCE REPORT",
        "bg": "#111827", "fg": "#e5e7eb", "accent": "#f59e0b",
        "header_bg": "#1e293b", "cell_bg": "#0f172a",
        "status_colors": {"good": "#22c55e", "warn": "#f59e0b", "bad": "#ef4444"},
        "entities": ["Rodriguez (#7)", "Chen (#14)", "Okafor (#22)",
                      "Petrov (#3)", "Silva (#9)", "Nakamura (#11)"],
        "attrs": ["Speed (km/h)", "Accuracy (%)", "Stamina"],
        "ranges": [(26, 37), (58, 96), (42, 98)],
        "memo_from": "Head Coach Martinez",
        "memo_re": "Game-Day Adjustments -- vs Dynamo FC (Away)",
        "filler": "Team bus departs 14:00 sharp. Away strip is navy alternate. "
                  "Post-match press: captain and one midfielder. "
                  "Physio reports no soft-tissue concerns. "
                  "Sections 14-16 for visiting supporters. "
                  "Video analysis of Dynamo's last three matches is available on the team tablet.",
    },
    "archaeological_survey": {
        "title": "SITE DX-7 EXCAVATION -- STRATIGRAPHY LOG",
        "bg": "#faf3e0", "fg": "#3d2b1f", "accent": "#8b4513",
        "header_bg": "#e0d5c0", "cell_bg": "#f0e8d5",
        "status_colors": {"good": "#6b8e23", "warn": "#cd853f", "bad": "#8b0000"},
        "entities": ["Layer A (0.5m)", "Layer B (1.2m)", "Layer C (2.0m)",
                      "Layer D (3.1m)", "Layer E (4.5m)", "Layer F (6.0m)"],
        "attrs": ["Artifacts", "Est. Age (yr)", "Integrity (%)"],
        "ranges": [(3, 42), (800, 7500), (15, 92)],
        "memo_from": "Lead Archaeologist Dr. Kowalski",
        "memo_re": "Field Corrections -- Carbon-14 Recalibration",
        "filler": "GPR scan suggests a chamber below Layer F. "
                  "Photography completed 360-degree documentation. "
                  "Site permit renewed through December. "
                  "Volunteer orientation Monday AM. "
                  "Soil moisture acceptable for continued excavation.",
    },
    "deep_ocean_sub": {
        "title": "SUBMERSIBLE 'HADAL-IX' -- DIVE TELEMETRY",
        "bg": "#051622", "fg": "#a8d8ea", "accent": "#00e5ff",
        "header_bg": "#0a2940", "cell_bg": "#071d30",
        "status_colors": {"good": "#00e5ff", "warn": "#ffab40", "bad": "#ff1744"},
        "entities": ["Ballast Tank", "Hull Sensor", "O2 Generator",
                      "Thruster Port", "Comm Relay", "Sonar Array"],
        "attrs": ["Depth (m)", "Pressure (atm)", "Temp (C)"],
        "ranges": [(50, 4000), (5, 400), (-2, 22)],
        "memo_from": "Dive Director Tanaka",
        "memo_re": "Depth Correction -- Dive #147 Calibration",
        "filler": "Bio-luminescence sampling scheduled for 3200m. "
                  "Robotic arm serviced; grip test nominal. "
                  "Acoustic beacon network realigned after current shift. "
                  "Specimen jars 7-12 sterilized and loaded. "
                  "Surface support vessel confirms stable sea state 2.",
    },
    "satellite_network": {
        "title": "ORBITAL CONSTELLATION -- SAT HEALTH REPORT",
        "bg": "#0c0c1e", "fg": "#b8c6db", "accent": "#4fc3f7",
        "header_bg": "#1a1a3e", "cell_bg": "#121230",
        "status_colors": {"good": "#4fc3f7", "warn": "#ffd54f", "bad": "#ef5350"},
        "entities": ["SAT-Alpha", "SAT-Beta", "SAT-Gamma",
                      "SAT-Delta", "SAT-Epsilon", "SAT-Zeta"],
        "attrs": ["Signal (dBm)", "Orbit (km)", "Power (W)"],
        "ranges": [(-120, -40), (200, 800), (50, 500)],
        "memo_from": "Mission Control Ops Lead Singh",
        "memo_re": "Link Budget Recalculation -- Eclipse Season",
        "filler": "Solar panel degradation within tolerance for year 3. "
                  "Debris avoidance maneuver for SAT-Gamma postponed. "
                  "Firmware v4.2 upload to constellation delayed to next window. "
                  "Ground station Svalbard back online after maintenance. "
                  "Spectrum coordination with ITU renewed for 12 months.",
    },
    "air_traffic_control": {
        "title": "ATC SECTOR 7-WEST -- TRAFFIC STATE",
        "bg": "#0a1a0a", "fg": "#b0e0b0", "accent": "#39ff14",
        "header_bg": "#1a3a1a", "cell_bg": "#0f250f",
        "status_colors": {"good": "#39ff14", "warn": "#ffd700", "bad": "#ff073a"},
        "entities": ["Flight AA-412", "Flight DL-891", "Flight UA-223",
                      "Flight BA-117", "Flight LH-654", "Flight QF-888"],
        "attrs": ["Alt (ft)", "Speed (kts)", "Fuel (kg)"],
        "ranges": [(5000, 41000), (180, 550), (2000, 18000)],
        "memo_from": "Sector Supervisor Collins",
        "memo_re": "Routing Directive -- Severe Weather Avoidance",
        "filler": "Runway 28L at destination closed for resurfacing. "
                  "SIGMET for CB activity FL250-FL390 sector 9-East. "
                  "Military exercise airspace CHARLIE active until 2200Z. "
                  "ATIS Bravo current at destination. "
                  "Relief controller reports at 1800Z for handover.",
    },
    "cybersecurity_soc": {
        "title": "THREAT INTEL DASHBOARD -- INCIDENT LOG Q2",
        "bg": "#1a0a0a", "fg": "#e0b0b0", "accent": "#ff4444",
        "header_bg": "#2a1515", "cell_bg": "#1f0f0f",
        "status_colors": {"good": "#66bb6a", "warn": "#ffa726", "bad": "#ef5350"},
        "entities": ["CVE-2026-1042", "CVE-2026-1137", "CVE-2026-1298",
                      "CVE-2026-1415", "CVE-2026-1507", "CVE-2026-1621"],
        "attrs": ["Severity (1-10)", "Hosts Hit", "Response (hrs)"],
        "ranges": [(1, 10), (1, 500), (1, 72)],
        "memo_from": "CISO Rebecca Torres",
        "memo_re": "Incident Severity Re-assessment -- Post-Mortem",
        "filler": "Quarterly pen-test scheduled for May 15. "
                  "New SIEM rules deployed covering lateral movement. "
                  "Vendor patch for Exchange zero-day available; rollout Monday. "
                  "Security awareness training completion at 87%. "
                  "Board presentation on cyber risk posture next Thursday.",
    },
    "vineyard_harvest": {
        "title": "DOMAINE AURORE -- HARVEST ANALYSIS 2026",
        "bg": "#1a0f1f", "fg": "#e8d5e8", "accent": "#c084fc",
        "header_bg": "#2d1a35", "cell_bg": "#1f1228",
        "status_colors": {"good": "#a78bfa", "warn": "#fbbf24", "bad": "#f87171"},
        "entities": ["Pinot Noir A", "Chardonnay B", "Merlot C",
                      "Syrah D", "Riesling E", "Cab Sauv F"],
        "attrs": ["Brix", "Acidity (g/L)", "Yield (t/ha)"],
        "ranges": [(18, 28), (4, 9), (2, 12)],
        "memo_from": "Head Winemaker Dubois",
        "memo_re": "Pre-Crush Corrections -- Refractometer Calibration",
        "filler": "Fermentation tanks 4-8 cleaned and sulfited. "
                  "Oak barrel allocation for reserve program finalized. "
                  "Forecast shows no rain for the next 10 days — ideal for extended hang time. "
                  "Seasonal harvest crew arrives Wednesday at 06:00. "
                  "Label design for the 2026 vintage approved by marketing.",
    },
}

# =====================================================================
# IMAGE RENDERER (Enhanced with status indicators + decoy panels)
# =====================================================================

def get_status(val, lo, hi):
    """Compute a status label from value position in range."""
    frac = (val - lo) / (hi - lo) if hi != lo else 0.5
    if frac > 0.7:
        return "HIGH"
    elif frac > 0.3:
        return "MID"
    return "LOW"

def get_status_color(status, colors):
    return {"HIGH": colors["bad"], "MID": colors["warn"], "LOW": colors["good"]}.get(status, colors["warn"])

def format_val(v):
    return str(v) if isinstance(v, int) else f"{v:.1f}"

def render_image(domain_name, display_data, sample_id, extra_panels=None):
    cfg = DOMAINS[domain_name]
    entities = cfg["entities"]
    attrs = cfg["attrs"]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.patch.set_facecolor(cfg["bg"])
    ax.set_facecolor(cfg["bg"])
    ax.axis("off")

    # Title with decorative line
    ax.set_title(cfg["title"], color=cfg["accent"],
                 fontsize=13, fontweight="bold", pad=20, fontfamily="monospace")

    # Build table with status column
    col_labels = [""] + attrs + ["Status"]
    rows = []
    cell_colors = []
    for entity in entities:
        vals = display_data[entity]
        lo, hi = cfg["ranges"][0]
        status = get_status(vals[0], lo, hi)
        row = [entity] + [format_val(v) for v in vals] + [status]
        rows.append(row)

        row_colors = [cfg["cell_bg"]] * (len(attrs) + 1)
        row_colors.append(cfg["cell_bg"])
        cell_colors.append(row_colors)

    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.85)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(cfg["header_bg"])
        cell.set_text_props(color=cfg["accent"], fontweight="bold", fontsize=9)
        cell.set_edgecolor(cfg["header_bg"])

    # Style body
    for i in range(len(rows)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_facecolor(cfg["cell_bg"])
            cell.set_edgecolor(cfg["header_bg"])
            if j == 0:
                cell.set_text_props(color=cfg["accent"], fontweight="bold", fontsize=9)
            elif j == len(col_labels) - 1:
                # Status column with color
                status = rows[i][-1]
                sc = get_status_color(status, cfg["status_colors"])
                cell.set_text_props(color=sc, fontweight="bold", fontsize=9)
            else:
                cell.set_text_props(color=cfg["fg"], fontsize=10)

    # Decoy panels (visual distractors)
    if extra_panels:
        for idx, panel in enumerate(extra_panels):
            y_pos = 0.02 + idx * 0.05
            fig.text(0.5, y_pos, panel["text"],
                     ha="center", va="bottom", fontsize=9,
                     color=cfg["accent"], fontfamily="monospace", alpha=0.85,
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor=cfg["header_bg"],
                               edgecolor=cfg["accent"], alpha=0.7))

    img_path = IMAGES_DIR / f"{sample_id}.png"
    plt.savefig(img_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return f"images/{sample_id}.png"


# =====================================================================
# TEXT GENERATION HELPERS
# =====================================================================

def build_memo_header(cfg):
    return [f"MEMO: {cfg['memo_re']}", f"From: {cfg['memo_from']}", ""]

def build_corrections(corruptions, entities, attrs):
    lines = []
    for ei, ai, dv, tv in corruptions:
        lines.append(f"CORRECTION: {entities[ei]}'s {attrs[ai]} is actually "
                     f"{format_val(tv)}, not {format_val(dv)} as displayed.")
    return lines

def build_filler(cfg, conflict):
    if conflict < 4:
        return []
    lines = ["", "--- Other Notes ---", cfg["filler"]]
    if conflict >= 5:
        lines += ["", "The above notes are for general awareness only and do not affect "
                  "any corrections or formulas in this memo."]
    return lines

def build_c5_contradiction(corruptions, entities, attrs):
    """C5: Add a misleading 'earlier correction' that is then overridden."""
    if not corruptions:
        return []
    ei, ai, dv, tv = corruptions[0]
    fake_val = round(tv * random.choice([0.8, 0.9, 1.1, 1.2]), 1)
    return [
        f"PRELIMINARY NOTE: An earlier audit suggested {entities[ei]}'s {attrs[ai]} "
        f"might be {format_val(fake_val)}, but this was superseded by the correction above.",
    ]


# =====================================================================
# CORRUPTION GENERATOR
# =====================================================================

def make_corruptions(true_data, entities, attrs, count):
    corruptions = []
    used = set()
    indices = list(range(len(entities)))
    random.shuffle(indices)
    for k in range(min(count, len(entities))):
        ei = indices[k]
        ai = random.randint(0, len(attrs) - 1)
        tv = true_data[entities[ei]][ai]
        if isinstance(tv, int):
            offset = random.choice([-1, 1]) * random.randint(2, max(3, abs(tv) // 4 + 1))
            dv = max(1, tv + offset)
        else:
            offset = random.choice([-1, 1]) * round(random.uniform(0.5, max(1, abs(tv) * 0.15)), 1)
            dv = round(max(0.1, tv + offset), 1)
        corruptions.append((ei, ai, dv, tv))
        used.add((ei, ai))
    return corruptions, used


# =====================================================================
# Q&A GENERATORS (I1-I5) -- Enhanced difficulty
# =====================================================================

def pick_target(entities, attrs, exclude=None):
    exclude = exclude or set()
    attempts = 0
    while attempts < 100:
        ei = random.randint(0, len(entities) - 1)
        ai = random.randint(0, len(attrs) - 1)
        if (ei, ai) not in exclude:
            return ei, ai
        attempts += 1
    return 0, 0


def qa_i1(cfg, true_data, display_data, entities, attrs, corrupted):
    """I1: Apply a percentage adjustment from text to one image value."""
    ei, ai = pick_target(entities, attrs)
    entity, attr = entities[ei], attrs[ai]
    pct = random.choice([5, 8, 10, 12, 15, -5, -8, -10, -12, -15])
    factor = 1 + pct / 100
    tv = true_data[entity][ai]
    dv = display_data[entity][ai]
    answer = round(tv * factor, 1)
    trap = round(dv * factor, 1) if (ei, ai) in corrupted else round(dv, 1)
    sign = "+" if pct > 0 else ""

    rule = (f"ADJUSTMENT: Apply a {sign}{pct}% correction to {entity}'s {attr}. "
            f"Use the corrected value from above if one was provided.")
    question = (f"After applying the memo's percentage adjustment (and any corrections), "
                f"what is {entity}'s final {attr}? Reply with the number only.")
    rationale = (f"True {entity} {attr} = {format_val(tv)}. "
                 f"Factor = {factor}. Answer = {format_val(tv)} x {factor} = {format_val(answer)}.")
    return question, answer, trap, rule, rationale


def qa_i2(cfg, true_data, display_data, entities, attrs, corrupted):
    """I2: Compute (attr_A - attr_B) using text formula."""
    ei = random.randint(0, len(entities) - 1)
    ai1, ai2 = 0, 1
    entity = entities[ei]
    tv1, tv2 = true_data[entity][ai1], true_data[entity][ai2]
    dv1, dv2 = display_data[entity][ai1], display_data[entity][ai2]
    answer = round(tv1 - tv2, 1)
    trap = round(dv1 - dv2, 1)

    rule = (f"FORMULA: Compute the Differential for any entity as "
            f"({attrs[ai1]} minus {attrs[ai2]}). Use corrected values.")
    question = (f"Using the memo's formula (with corrected data), "
                f"what is {entity}'s Differential ({attrs[ai1]} - {attrs[ai2]})? Number only.")
    rationale = (f"True {attrs[ai1]} = {format_val(tv1)}, True {attrs[ai2]} = {format_val(tv2)}. "
                 f"Answer = {format_val(tv1)} - {format_val(tv2)} = {format_val(answer)}.")
    return question, answer, trap, rule, rationale


def qa_i3(cfg, true_data, display_data, entities, attrs, corrupted):
    """I3: Conditional -- if entity_A meets condition, adjust entity_B."""
    eiA = random.randint(0, len(entities) - 1)
    eiB = (eiA + random.randint(1, 3)) % len(entities)
    ai_cond = random.randint(0, len(attrs) - 1)
    ai_target = random.randint(0, len(attrs) - 1)
    entityA, entityB = entities[eiA], entities[eiB]
    val_cond = true_data[entityA][ai_cond]
    mid = (cfg["ranges"][ai_cond][0] + cfg["ranges"][ai_cond][1]) / 2
    threshold = round(mid, 1) if isinstance(mid, float) else int(mid)
    cond_met = val_cond > threshold
    target_val = true_data[entityB][ai_target]
    disp_target = display_data[entityB][ai_target]
    bonus = random.choice([5, 8, 10, 15, 20])
    answer = round(target_val + bonus, 1) if cond_met else round(target_val, 1)
    trap = round(disp_target + bonus, 1)

    rule = (f"CONDITIONAL: If {entityA}'s {attrs[ai_cond]} exceeds {format_val(threshold)}, "
            f"add {bonus} to {entityB}'s {attrs[ai_target]}. Otherwise leave unchanged. "
            f"Use corrected values.")
    question = (f"Following the conditional rule (with corrected data), "
                f"what is {entityB}'s effective {attrs[ai_target]}? Number only.")
    cond_s = "met" if cond_met else "NOT met"
    rationale = (f"{entityA} {attrs[ai_cond]} = {format_val(val_cond)}, "
                 f"threshold = {format_val(threshold)}, condition {cond_s}. "
                 f"{entityB} {attrs[ai_target]} = {format_val(target_val)}. "
                 f"Answer = {format_val(answer)}.")
    return question, answer, trap, rule, rationale


def qa_i4(cfg, true_data, display_data, entities, attrs, corrupted):
    """I4: Dual-condition filter + weighted aggregate (HARDER than v1)."""
    ai_f1 = 0
    ai_f2 = 2
    ai_sum = 1
    mid1 = (cfg["ranges"][ai_f1][0] + cfg["ranges"][ai_f1][1]) / 2
    mid2 = (cfg["ranges"][ai_f2][0] + cfg["ranges"][ai_f2][1]) / 2
    t1 = round(mid1, 1)
    t2 = round(mid2, 1)
    weight = random.choice([1.1, 1.15, 1.2, 0.9, 0.85])

    qualifying = []
    total_true = 0.0
    total_disp = 0.0
    for entity in entities:
        if true_data[entity][ai_f1] > t1 and true_data[entity][ai_f2] > t2:
            qualifying.append(entity)
            total_true += true_data[entity][ai_sum]
            total_disp += display_data[entity][ai_sum]

    if not qualifying:
        t1 = cfg["ranges"][ai_f1][0]
        t2 = cfg["ranges"][ai_f2][0]
        for entity in entities:
            if true_data[entity][ai_f1] > t1 and true_data[entity][ai_f2] > t2:
                qualifying.append(entity)
                total_true += true_data[entity][ai_sum]
                total_disp += display_data[entity][ai_sum]

    answer = round(total_true * weight, 1)
    trap = round(total_disp * weight, 1)
    w_pct = int((weight - 1) * 100) if weight > 1 else int((1 - weight) * -100)
    w_sign = "+" if w_pct >= 0 else ""

    rule = (f"AGGREGATION: Sum {attrs[ai_sum]} for entities where BOTH "
            f"{attrs[ai_f1]} > {format_val(t1)} AND {attrs[ai_f2]} > {format_val(t2)}. "
            f"Then apply a {w_sign}{w_pct}% weight to the total. Use corrected values.")
    question = (f"Per the aggregation rule (corrected data), "
                f"what is the weighted total {attrs[ai_sum]}? Number only.")
    rationale = (f"Qualifying (both conditions): {', '.join(qualifying) if qualifying else 'none'}. "
                 f"Raw sum = {format_val(round(total_true, 1))}. "
                 f"x {weight} = {format_val(answer)}.")
    return question, answer, trap, rule, rationale


def qa_i5(cfg, true_data, display_data, entities, attrs, corrupted):
    """I5: Multi-hop with cross-entity reference + branching (HARDER)."""
    ai0, ai1, ai2 = 0, 1, 2

    # Step 1: entity with highest attr[0]
    best = max(entities, key=lambda e: true_data[e][ai0])
    best_v0 = true_data[best][ai0]

    # Step 2: entity with lowest attr[0]
    worst = min(entities, key=lambda e: true_data[e][ai0])
    worst_v2 = true_data[worst][ai2]

    # Step 3: compute: best's attr[1] * factor - worst's attr[2]
    adj_pct = random.choice([10, 15, 20, 25])
    best_v1 = true_data[best][ai1]
    computed = round(best_v1 * (1 + adj_pct / 100) - worst_v2, 1)

    # Step 4: classify
    class_t = round((cfg["ranges"][ai1][0] + cfg["ranges"][ai1][1]) / 2, 1)
    label = "CRITICAL" if computed > class_t else "STABLE"

    # Trap from display data
    d_best = max(entities, key=lambda e: display_data[e][ai0])
    d_worst = min(entities, key=lambda e: display_data[e][ai0])
    d_computed = round(display_data[d_best][ai1] * (1 + adj_pct / 100) - display_data[d_worst][ai2], 1)
    trap = "CRITICAL" if d_computed > class_t else "STABLE"

    rule = (f"MULTI-STEP PROCEDURE:\n"
            f"  Step 1: Find the entity with the HIGHEST {attrs[ai0]} (corrected).\n"
            f"  Step 2: Find the entity with the LOWEST {attrs[ai0]} (corrected).\n"
            f"  Step 3: Compute: (Step-1 entity's {attrs[ai1]} x {1 + adj_pct/100}) "
            f"minus (Step-2 entity's {attrs[ai2]}).\n"
            f"  Step 4: If result > {format_val(class_t)} -> 'CRITICAL', else -> 'STABLE'.")
    question = (f"Follow the multi-step procedure (corrected data). "
                f"What is the classification: CRITICAL or STABLE?")
    rationale = (f"Step 1: highest {attrs[ai0]} -> {best} ({format_val(best_v0)}). "
                 f"Step 2: lowest {attrs[ai0]} -> {worst}. "
                 f"Step 3: {format_val(best_v1)} x {1+adj_pct/100} - {format_val(worst_v2)} = {format_val(computed)}. "
                 f"Step 4: {format_val(computed)} {'>' if label=='CRITICAL' else '<='} "
                 f"{format_val(class_t)} -> {label}.")
    return question, label, trap, rule, rationale


# =====================================================================
# SAMPLE GENERATOR
# =====================================================================

def generate_sample(sample_id, domain_name, integration, conflict):
    cfg = DOMAINS[domain_name]
    entities = cfg["entities"]
    attrs = cfg["attrs"]

    # 1. TRUE DATA
    true_data = {}
    for entity in entities:
        vals = []
        for j, attr in enumerate(attrs):
            lo, hi = cfg["ranges"][j]
            if any(k in attr.lower() for k in ["artifact", "building", "host"]):
                vals.append(random.randint(int(lo), int(hi)))
            else:
                vals.append(round(random.uniform(lo, hi), 1))
        true_data[entity] = vals

    # 2. CORRUPTIONS
    num_corrupt = min(conflict - 1, 4)  # C1=0, C2=1, C3=2, C4=3, C5=4
    corruptions, corrupted_cells = make_corruptions(true_data, entities, attrs, num_corrupt)
    display_data = {e: list(v) for e, v in true_data.items()}
    for ei, ai, dv, _ in corruptions:
        display_data[entities[ei]][ai] = dv

    # 3. TEXT
    text_parts = build_memo_header(cfg)
    text_parts += build_corrections(corruptions, entities, attrs)
    if conflict >= 5:
        text_parts += build_c5_contradiction(corruptions, entities, attrs)

    # 4. Q&A
    qa_fn = {1: qa_i1, 2: qa_i2, 3: qa_i3, 4: qa_i4, 5: qa_i5}[integration]
    q, a, trap, rule_text, rationale = qa_fn(cfg, true_data, display_data, entities, attrs, corrupted_cells)
    text_parts += ["", rule_text]
    text_parts += build_filler(cfg, conflict)

    # 5. DECOY PANELS
    extra_panels = []
    if conflict >= 3:
        de = random.randint(0, len(entities) - 1)
        da = random.randint(0, len(attrs) - 1)
        dv = round(true_data[entities[de]][da] * random.choice([1.2, 0.75, 1.35, 0.65]), 1)
        extra_panels.append({"text": f">> SUMMARY: {entities[de]} {attrs[da]} ~ {format_val(dv)}  [est.]"})
        text_parts.insert(3, "NOTE: Ignore estimated summary stats on the dashboard; use only the main table and corrections.")
    if conflict >= 5:
        de2 = random.randint(0, len(entities) - 1)
        da2 = random.randint(0, len(attrs) - 1)
        dv2 = round(true_data[entities[de2]][da2] * random.choice([0.5, 1.5]), 1)
        extra_panels.append({"text": f">> AVG {attrs[da2]}: {format_val(dv2)}  [projected]"})

    # 6. RENDER
    img_rel = render_image(domain_name, display_data, sample_id, extra_panels or None)
    text = "\n".join(text_parts)

    return {
        "id": sample_id,
        "image": img_rel,
        "text_passage": text,
        "question": q,
        "correct_answer": str(a),
        "image_only_trap": str(trap),
        "domain": domain_name,
        "integration_depth": integration,
        "conflict_level": conflict,
        "difficulty_cell": f"I{integration}_C{conflict}",
        "num_corruptions": num_corrupt,
        "text_length_chars": len(text),
        "requires_arithmetic": integration >= 2,
        "requires_conditional": integration >= 3,
        "requires_aggregation": integration >= 4,
        "requires_multihop": integration == 5,
        "rationale": rationale,
    }


# =====================================================================
# MAIN GENERATION LOOP
# =====================================================================

def generate_all(n_per_domain=50):
    domain_names = list(DOMAINS.keys())
    all_samples = []
    counter = 0

    for dname in domain_names:
        combos = [(i, c) for i in range(1, 6) for c in range(1, 6)]  # 25 cells
        random.shuffle(combos)
        # 2 samples per cell = 50 per domain
        full = combos + combos  # duplicate for 2 per cell
        full = sorted(full[:n_per_domain])

        for integration, conflict in full:
            counter += 1
            sid = f"cmat_{counter:04d}"
            print(f"  [{counter:3d}/500] {sid}  {dname:<22} I{integration}_C{conflict}", flush=True)
            sample = generate_sample(sid, dname, integration, conflict)
            all_samples.append(sample)

    return all_samples


def write_metadata(samples):
    out = SCRIPT_DIR / "metadata.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n[OK] Wrote {len(samples)} records to {out}")


def write_dataset_card(n_samples):
    card = textwrap.dedent(f"""\
    # Cross-Modal Attention Triage (CMAT) Dataset v2.0

    ## Overview
    Benchmark for multimodal LLM cross-modal attention: can the model
    integrate both a TEXT passage AND an IMAGE to produce a correct answer?
    Every sample is designed so that NEITHER modality alone is sufficient.

    ## Domains (10)
    | # | Domain | Theme |
    |---|--------|-------|
    | 1 | Space Mission Control | Sensor telemetry + protocol overrides |
    | 2 | Alchemy Lab | Reagent properties + recipe rules |
    | 3 | City Planning Board | District stats + regulation memos |
    | 4 | Sports Analytics | Player metrics + coach adjustments |
    | 5 | Archaeological Survey | Excavation data + field corrections |
    | 6 | Deep Ocean Submersible | Dive telemetry + emergency protocols |
    | 7 | Satellite Network | Orbital health + comms priority rules |
    | 8 | Air Traffic Control | Flight metrics + routing directives |
    | 9 | Cybersecurity SOC | Threat indicators + incident response |
    | 10 | Vineyard Harvest | Grape metrics + winemaking rules |

    ## Difficulty Axes
    **Integration Depth (I1-I5)** -- cross-modal reasoning complexity:
    - I1: single text adjustment applied to one image value
    - I2: text formula applied to two image values
    - I3: conditional rule (text if-then checked against image)
    - I4: dual-condition filter + weighted aggregate across entities
    - I5: multi-hop chain with cross-entity reference + classification

    **Conflict Level (C1-C5)** -- input noise and misdirection:
    - C1: no corruption, clean text
    - C2: 1 corrupted image value, text correction
    - C3: 2 corruptions + decoy panel + ignore instruction
    - C4: 3 corruptions + long filler text burying corrections
    - C5: 4 corruptions + contradictory preliminary notes + dual decoy panels

    ## Statistics
    - Total samples: {n_samples}
    - Domains: 10 x {n_samples // 10} each
    - Difficulty grid: 25 cells (I1-I5 x C1-C5) x 2 per domain
    - 100% synthetic, zero real-world data, fully reproducible

    ## Schema (metadata.jsonl)
    | Field | Type | Description |
    |-------|------|-------------|
    | id | str | Sample ID (cmat_NNNN) |
    | image | str | Relative path to PNG image |
    | text_passage | str | The memo/text input |
    | question | str | Question requiring both modalities |
    | correct_answer | str | Ground-truth answer |
    | image_only_trap | str | Plausible wrong answer from image alone |
    | domain | str | Domain name |
    | integration_depth | int | 1-5 |
    | conflict_level | int | 1-5 |
    | difficulty_cell | str | e.g. I3_C2 |
    | rationale | str | Step-by-step solution chain |
    """)
    p = SCRIPT_DIR / "dataset_card.md"
    with open(p, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"[OK] Wrote dataset card to {p}")


if __name__ == "__main__":
    print("=" * 60)
    print("  CMAT Dataset Generator v2.0 -- FINAL")
    print("  10 Domains x 50 Samples = 500 Total")
    print("=" * 60, flush=True)

    samples = generate_all(n_per_domain=50)
    write_metadata(samples)
    write_dataset_card(len(samples))

    print()
    print("=" * 60)
    print(f"  GENERATION COMPLETE")
    print(f"  Total: {len(samples)} samples")
    print(f"  Images: {IMAGES_DIR}")
    print(f"  Metadata: {SCRIPT_DIR / 'metadata.jsonl'}")
    print("=" * 60)

    from collections import Counter
    dc = Counter(s["domain"] for s in samples)
    ic = Counter(s["difficulty_cell"] for s in samples)
    print("\n  Per domain:")
    for d, c in sorted(dc.items()):
        print(f"    {d:<25} {c}")
    print("\n  Per difficulty cell:")
    for d, c in sorted(ic.items()):
        print(f"    {d:<10} {c}")
