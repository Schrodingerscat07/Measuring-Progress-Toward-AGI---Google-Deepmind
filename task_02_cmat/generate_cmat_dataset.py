#!/usr/bin/env python3
"""
CMAT Dataset Generator v1.0 — Cross-Modal Attention Triage
==========================================================
100 samples across 5 creative domains × 4 integration depths × varying conflict.
Each sample: PNG image + text passage + question + deterministic answer.
The answer REQUIRES integrating information from BOTH modalities.

Domains:
  1. Space Mission Control   — sensor telemetry + protocol overrides
  2. Alchemy Lab             — ingredient properties + recipe rules
  3. City Planning Board     — district statistics + regulation memos
  4. Sports Analytics        — player metrics + coach adjustments
  5. Archaeological Survey   — excavation findings + field note corrections

Usage:  python generate_cmat_dataset.py
Output: task_02_cmat/images/*.png + task_02_cmat/metadata.jsonl
"""

import os
import json
import random
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Reproducibility ──
random.seed(42)
np.random.seed(42)

# ── Paths ──
SCRIPT_DIR = Path(__file__).parent
IMAGES_DIR = SCRIPT_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# DOMAIN CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════

DOMAINS = {
    "space_mission": {
        "title": "DEEP SPACE VESSEL 'AURORA' — SENSOR ARRAY",
        "bg": "#0a0e27", "fg": "#c8d6e5", "accent": "#00ff88",
        "header_bg": "#162044", "cell_bg": "#0f1535",
        "entities": [
            "Thermal Array", "Pressure Hull", "O₂ Recycler",
            "Fuel Cell", "Radiation Shield", "Nav Beacon",
        ],
        "attrs": ["Reading", "Baseline", "Drift (%)"],
        "ranges": [(-15, 75), (85, 125), (-8, 12)],
        "memo_from": "Chief Engineer Vasquez",
        "memo_re": "Sensor Calibration Override — Cycle 47",
        "filler": (
            "All crew: routine maintenance window for Deck 7 has been rescheduled to 0300 UTC. "
            "Hydroponics bay reports nominal crop yield. "
            "External hull inspection by EVA team completed with no anomalies noted. "
            "Mess hall protein recycler back online after the Tuesday fault. "
            "Next port-of-call ETA remains 14.2 standard days."
        ),
    },
    "alchemy_lab": {
        "title": "GRAND ALCHEMIST'S WORKBENCH — REAGENT LOG",
        "bg": "#1a1207", "fg": "#e8d5a3", "accent": "#ff9900",
        "header_bg": "#2d2010", "cell_bg": "#1f1809",
        "entities": [
            "Dragon Petal", "Moon Salt", "Wyrm Venom",
            "Starlight Dew", "Iron Bloom", "Ghost Moss",
        ],
        "attrs": ["Volume (mL)", "Purity (%)", "Potency"],
        "ranges": [(10, 195), (55, 98), (1, 10)],
        "memo_from": "Archmage Elara",
        "memo_re": "Recipe Corrections — Elixir of Clarity, Batch 12",
        "filler": (
            "The apprentice council has voted to extend brewing hours on Wednesdays. "
            "Cauldron #4 in the east wing requires re-seasoning — do not use until further notice. "
            "New shipment of crystal vials arrives next Moonday. "
            "Reminder: all reagent spills must be reported to the Safety Warlock immediately. "
            "Annual familiar vaccination drive begins next tenday."
        ),
    },
    "city_planning": {
        "title": "METRO COUNCIL — DISTRICT STATISTICS Q2 2026",
        "bg": "#f5f0e8", "fg": "#2c3e50", "accent": "#2980b9",
        "header_bg": "#d5cfc3", "cell_bg": "#ece7db",
        "entities": [
            "Riverside Ward", "Northgate Hub", "Old Quarter",
            "Tech Corridor", "Harbor District", "Greenfield Zone",
        ],
        "attrs": ["Pop. (K)", "Buildings", "Green Cover (%)"],
        "ranges": [(8, 115), (30, 480), (5, 42)],
        "memo_from": "City Planner Dr. Okonkwo",
        "memo_re": "Quarterly Zoning Corrections & Projections",
        "filler": (
            "The public hearing for the waterfront promenade received 312 comments, mostly positive. "
            "Bus route 14B will be extended to Greenfield starting June. "
            "Heritage preservation applications for Old Quarter are under review. "
            "The new cycling infrastructure proposal passed committee 5-2. "
            "Next council meeting: April 28th, 7 PM, City Hall Room B."
        ),
    },
    "sports_analytics": {
        "title": "THUNDER FC — PRE-MATCH PERFORMANCE REPORT",
        "bg": "#111827", "fg": "#e5e7eb", "accent": "#f59e0b",
        "header_bg": "#1e293b", "cell_bg": "#0f172a",
        "entities": [
            "Rodriguez (#7)", "Chen (#14)", "Okafor (#22)",
            "Petrov (#3)", "Silva (#9)", "Nakamura (#11)",
        ],
        "attrs": ["Speed (km/h)", "Accuracy (%)", "Stamina"],
        "ranges": [(26, 37), (58, 96), (42, 98)],
        "memo_from": "Head Coach Martinez",
        "memo_re": "Game-Day Adjustments — vs. Dynamo FC (Away)",
        "filler": (
            "Team bus departs at 14:00 sharp from the training ground gate. "
            "Kit manager confirms the away strip will be the navy alternate. "
            "Post-match press conference assigned to the captain and one midfielder. "
            "Physio reports no new soft-tissue concerns from Thursday's session. "
            "Fan travel advisory: sections 14-16 are allocated to visiting supporters."
        ),
    },
    "archaeological_survey": {
        "title": "SITE ΔX-7 EXCAVATION — STRATIGRAPHY LOG",
        "bg": "#faf3e0", "fg": "#3d2b1f", "accent": "#8b4513",
        "header_bg": "#e0d5c0", "cell_bg": "#f0e8d5",
        "entities": [
            "Layer α (0.5 m)", "Layer β (1.2 m)", "Layer γ (2.0 m)",
            "Layer δ (3.1 m)", "Layer ε (4.5 m)", "Layer ζ (6.0 m)",
        ],
        "attrs": ["Artifacts", "Est. Age (yr)", "Integrity (%)"],
        "ranges": [(3, 42), (800, 7500), (15, 92)],
        "memo_from": "Lead Archaeologist Dr. Kowalski",
        "memo_re": "Field Corrections — Carbon-14 Recalibration",
        "filler": (
            "The new ground-penetrating radar scan suggests a possible chamber below Layer ζ. "
            "Photography team has completed full 360° documentation of the main trench. "
            "Local site permit renewed through December. "
            "Volunteer orientation for the university cohort is scheduled for Monday AM. "
            "Soil moisture levels remain acceptable for continued excavation."
        ),
    },
}

# ═══════════════════════════════════════════════════════════════════
# IMAGE RENDERER
# ═══════════════════════════════════════════════════════════════════

def render_image(domain_name: str, display_data: dict, sample_id: str,
                 extra_panel: dict | None = None) -> str:
    """Render a styled data-table image. Returns the image path."""
    cfg = DOMAINS[domain_name]
    entities = cfg["entities"]
    attrs = cfg["attrs"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(cfg["bg"])
    ax.set_facecolor(cfg["bg"])
    ax.axis("off")
    ax.set_title(cfg["title"], color=cfg["accent"],
                 fontsize=13, fontweight="bold", pad=18, fontfamily="monospace")

    # Build table data
    col_labels = [""] + attrs
    rows = []
    for entity in entities:
        vals = display_data[entity]
        row = [entity] + [format_val(v) for v in vals]
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.9)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(cfg["header_bg"])
        cell.set_text_props(color=cfg["accent"], fontweight="bold", fontsize=10)
        cell.set_edgecolor(cfg["header_bg"])
    # Style body
    for i in range(len(rows)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_facecolor(cfg["cell_bg"])
            cell.set_text_props(color=cfg["fg"], fontsize=11)
            cell.set_edgecolor(cfg["header_bg"])
            if j == 0:
                cell.set_text_props(color=cfg["accent"], fontweight="bold", fontsize=10)

    # Optional extra panel (decoy at higher difficulty)
    if extra_panel:
        fig.text(0.5, 0.02, extra_panel["text"],
                 ha="center", va="bottom", fontsize=10,
                 color=cfg["accent"], fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4",
                           facecolor=cfg["header_bg"], edgecolor=cfg["accent"]))

    img_path = IMAGES_DIR / f"{sample_id}.png"
    plt.savefig(img_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return f"task_02_cmat/images/{sample_id}.png"


def format_val(v):
    if isinstance(v, int):
        return str(v)
    return f"{v:.1f}"


# ═══════════════════════════════════════════════════════════════════
# TEXT + QUESTION + ANSWER GENERATION
# ═══════════════════════════════════════════════════════════════════

def pick_target(entities, attrs, exclude=None):
    """Pick a random (entity_idx, attr_idx) avoiding exclude set."""
    exclude = exclude or set()
    while True:
        ei = random.randint(0, len(entities) - 1)
        ai = random.randint(0, len(attrs) - 1)
        if (ei, ai) not in exclude:
            return ei, ai


def make_corruptions(true_data, entities, attrs, count):
    """Create 'count' corrupted cells with plausible wrong values."""
    corruptions = []
    used = set()
    indices = list(range(len(entities)))
    random.shuffle(indices)
    for k in range(min(count, len(entities))):
        ei = indices[k]
        ai = random.randint(0, len(attrs) - 1)
        true_val = true_data[entities[ei]][ai]
        if isinstance(true_val, int):
            offset = random.choice([-1, 1]) * random.randint(2, max(3, abs(true_val) // 4 + 1))
            dv = max(1, true_val + offset)
        else:
            offset = random.choice([-1, 1]) * round(random.uniform(0.5, max(1, abs(true_val) * 0.15)), 1)
            dv = round(max(0.1, true_val + offset), 1)
        corruptions.append((ei, ai, dv, true_val))
        used.add((ei, ai))
    return corruptions, used


def build_memo_header(cfg, conflict):
    lines = [
        f"MEMO: {cfg['memo_re']}",
        f"From: {cfg['memo_from']}",
        "",
    ]
    return lines


def build_corrections_text(corruptions, entities, attrs):
    lines = []
    for ei, ai, disp_val, true_val in corruptions:
        lines.append(
            f"CORRECTION: {entities[ei]}'s {attrs[ai]} is actually "
            f"{format_val(true_val)}, not {format_val(disp_val)} as displayed."
        )
    return lines


def build_filler(cfg, conflict):
    if conflict < 4:
        return []
    return ["", "--- Other Notes ---", cfg["filler"], ""]


def generate_sample(sample_id, domain_name, integration, conflict):
    """Generate one complete CMAT sample."""
    cfg = DOMAINS[domain_name]
    entities = cfg["entities"]
    attrs = cfg["attrs"]

    # ── 1. Generate TRUE data ──
    true_data = {}
    for entity in entities:
        vals = []
        for j, attr in enumerate(attrs):
            lo, hi = cfg["ranges"][j]
            if any(k in attr.lower() for k in ["artifact", "building", "age"]):
                vals.append(random.randint(int(lo), int(hi)))
            else:
                vals.append(round(random.uniform(lo, hi), 1))
        true_data[entity] = vals

    # ── 2. Corrupt cells for display (conflict axis) ──
    num_corrupt = max(0, conflict - 1)  # C1=0, C2=1, C3=2, C4=3
    corruptions, corrupted_cells = make_corruptions(true_data, entities, attrs, num_corrupt)

    display_data = {e: list(v) for e, v in true_data.items()}
    for ei, ai, disp_val, _ in corruptions:
        display_data[entities[ei]][ai] = disp_val

    # ── 3. Build text passage ──
    text_parts = build_memo_header(cfg, conflict)
    text_parts += build_corrections_text(corruptions, entities, attrs)

    # ── 4. Generate Q&A based on integration depth ──
    if integration == 1:
        q, a, trap, rule_text, rationale = qa_i1(cfg, true_data, display_data, entities, attrs, corrupted_cells)
    elif integration == 2:
        q, a, trap, rule_text, rationale = qa_i2(cfg, true_data, display_data, entities, attrs, corrupted_cells)
    elif integration == 3:
        q, a, trap, rule_text, rationale = qa_i3(cfg, true_data, display_data, entities, attrs, corrupted_cells)
    elif integration == 4:
        q, a, trap, rule_text, rationale = qa_i4(cfg, true_data, display_data, entities, attrs, corrupted_cells)
    else:  # integration == 5
        q, a, trap, rule_text, rationale = qa_i5(cfg, true_data, display_data, entities, attrs, corrupted_cells)

    text_parts.append("")
    text_parts.append(rule_text)

    # Add filler at high conflict
    text_parts += build_filler(cfg, conflict)

    # ── 5. Decoy panel on image (C3+) ──
    extra_panel = None
    if conflict >= 3:
        # Add a misleading "summary" value on the image
        decoy_ei = random.randint(0, len(entities) - 1)
        decoy_ai = random.randint(0, len(attrs) - 1)
        decoy_v = round(true_data[entities[decoy_ei]][decoy_ai] * random.choice([1.2, 0.8, 1.35, 0.7]), 1)
        extra_panel = {
            "text": f">> QUICK STAT: {entities[decoy_ei]} {attrs[decoy_ai]} ~ {format_val(decoy_v)}  [est.]"
        }
        text_parts.insert(3, "NOTE: Ignore any estimated quick-stats shown on the dashboard panel; use only the main data table and these corrections.")

    # ── 6. Render image ──
    img_rel = render_image(domain_name, display_data, sample_id, extra_panel)

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


# ── Q&A Generators per Integration Level ──


def qa_i1(cfg, true_data, display_data, entities, attrs, corrupted):
    """I1: Apply a single text-provided adjustment to one image value."""
    ei, ai = pick_target(entities, attrs)
    entity, attr = entities[ei], attrs[ai]
    pct = random.choice([5, 8, 10, 12, 15, -5, -8, -10, -12, -15])
    factor = 1 + pct / 100
    true_val = true_data[entity][ai]
    disp_val = display_data[entity][ai]
    answer = round(true_val * factor, 1)
    trap = round(disp_val * factor, 1) if (ei, ai) in corrupted else round(disp_val, 1)

    sign = "+" if pct > 0 else ""
    rule = f"ADJUSTMENT: Apply a {sign}{pct}% correction to {entity}'s {attr} (use the corrected value if one was provided above)."
    question = f"After applying the memo's percentage adjustment (and any corrections), what is {entity}'s final {attr}?"
    rationale = (
        f"True value of {entity} {attr} = {format_val(true_val)}. "
        f"Adjustment = {sign}{pct}%. "
        f"Answer: {format_val(true_val)} × {factor} = {format_val(answer)}."
    )
    return question, answer, trap, rule, rationale


def qa_i2(cfg, true_data, display_data, entities, attrs, corrupted):
    """I2: Compute (attr_A − attr_B) for a given entity using text formula."""
    ei = random.randint(0, len(entities) - 1)
    ai1, ai2 = 0, 1  # first two attrs
    entity = entities[ei]

    tv1 = true_data[entity][ai1]
    tv2 = true_data[entity][ai2]
    dv1 = display_data[entity][ai1]
    dv2 = display_data[entity][ai2]

    answer = round(tv1 - tv2, 1)
    trap = round(dv1 - dv2, 1)

    rule = (
        f"FORMULA: For each entity, compute the Differential as "
        f"({attrs[ai1]} minus {attrs[ai2]}). Use corrected values where applicable."
    )
    question = (
        f"Using the formula from the memo (and any corrections), "
        f"what is {entity}'s Differential ({attrs[ai1]} − {attrs[ai2]})?"
    )
    rationale = (
        f"True {attrs[ai1]} = {format_val(tv1)}, True {attrs[ai2]} = {format_val(tv2)}. "
        f"Answer: {format_val(tv1)} − {format_val(tv2)} = {format_val(answer)}."
    )
    return question, answer, trap, rule, rationale


def qa_i3(cfg, true_data, display_data, entities, attrs, corrupted):
    """I3: Conditional — if entity_A meets condition, compute entity_B's adjusted value."""
    eiA = random.randint(0, len(entities) - 1)
    eiB = (eiA + random.randint(1, 3)) % len(entities)
    ai_cond = random.randint(0, len(attrs) - 1)
    ai_target = random.randint(0, len(attrs) - 1)

    entityA, entityB = entities[eiA], entities[eiB]
    val_cond = true_data[entityA][ai_cond]

    # Set threshold so the condition is True ~50% of the time
    mid = (cfg["ranges"][ai_cond][0] + cfg["ranges"][ai_cond][1]) / 2
    threshold = round(mid, 1) if isinstance(mid, float) else int(mid)
    cond_met = val_cond > threshold

    target_val = true_data[entityB][ai_target]
    disp_target = display_data[entityB][ai_target]
    bonus = random.choice([5, 8, 10, 15, 20])

    if cond_met:
        answer = round(target_val + bonus, 1)
    else:
        answer = round(target_val, 1)

    trap = round(disp_target + bonus, 1)  # always applies bonus (wrong if condition not met)

    rule = (
        f"CONDITIONAL RULE: If {entityA}'s {attrs[ai_cond]} exceeds {format_val(threshold)}, "
        f"then add {bonus} to {entityB}'s {attrs[ai_target]}. Otherwise leave it unchanged. "
        f"Use corrected values for all checks."
    )
    question = (
        f"Following the conditional rule in the memo (using corrected data), "
        f"what is {entityB}'s effective {attrs[ai_target]}?"
    )
    cond_str = "met" if cond_met else "NOT met"
    rationale = (
        f"{entityA}'s {attrs[ai_cond]} = {format_val(val_cond)}. "
        f"Threshold = {format_val(threshold)}. Condition {cond_str}. "
        f"{entityB}'s {attrs[ai_target]} = {format_val(target_val)}. "
        f"Answer: {format_val(answer)}."
    )
    return question, answer, trap, rule, rationale


def qa_i4(cfg, true_data, display_data, entities, attrs, corrupted):
    """I4: Aggregate — sum attr values for entities exceeding a threshold in another attr."""
    ai_filter = random.randint(0, len(attrs) - 1)
    ai_sum = (ai_filter + 1) % len(attrs)

    mid = (cfg["ranges"][ai_filter][0] + cfg["ranges"][ai_filter][1]) / 2
    threshold = round(mid, 1) if isinstance(mid, float) else int(mid)

    qualifying = []
    total_true = 0
    total_disp = 0
    for entity in entities:
        if true_data[entity][ai_filter] > threshold:
            qualifying.append(entity)
            total_true += true_data[entity][ai_sum]
            total_disp += display_data[entity][ai_sum]

    # Fallback if no entities qualify
    if not qualifying:
        threshold = cfg["ranges"][ai_filter][0]
        for entity in entities:
            if true_data[entity][ai_filter] > threshold:
                qualifying.append(entity)
                total_true += true_data[entity][ai_sum]
                total_disp += display_data[entity][ai_sum]

    answer = round(total_true, 1)
    trap = round(total_disp, 1)

    rule = (
        f"AGGREGATION: Sum the {attrs[ai_sum]} across all entities whose "
        f"{attrs[ai_filter]} exceeds {format_val(threshold)}. Use corrected values."
    )
    question = (
        f"Per the aggregation rule in the memo (using corrected data), "
        f"what is the total {attrs[ai_sum]} for qualifying entities?"
    )
    rationale = (
        f"Entities with {attrs[ai_filter]} > {format_val(threshold)}: "
        f"{', '.join(qualifying)}. "
        f"Sum of their {attrs[ai_sum]}: {format_val(answer)}."
    )
    return question, answer, trap, rule, rationale


def qa_i5(cfg, true_data, display_data, entities, attrs, corrupted):
    """I5: Multi-hop — find entity with max attr → compute formula → classify."""
    ai_max = 0  # find max of first attr
    ai_second = 1  # use second attr of that entity

    # Step 1: find entity with highest attr[0]
    best_entity = max(entities, key=lambda e: true_data[e][ai_max])
    best_val = true_data[best_entity][ai_max]

    # Step 2: apply adjustment to its second attr
    adj_pct = random.choice([10, 15, 20, 25])
    second_val = true_data[best_entity][ai_second]
    adjusted = round(second_val * (1 + adj_pct / 100), 1)

    # Step 3: classify
    class_threshold = round((cfg["ranges"][ai_second][0] + cfg["ranges"][ai_second][1]) / 2, 1)
    label = "CRITICAL" if adjusted > class_threshold else "STABLE"

    # Trap: use display data instead of true data
    disp_best = max(entities, key=lambda e: display_data[e][ai_max])
    trap = "CRITICAL" if (display_data[disp_best][ai_second] * (1 + adj_pct / 100)) > class_threshold else "STABLE"

    rule = (
        f"MULTI-STEP PROCEDURE:\n"
        f"  Step 1: Identify the entity with the highest {attrs[ai_max]} (use corrected values).\n"
        f"  Step 2: Take that entity's {attrs[ai_second]} and increase it by {adj_pct}%.\n"
        f"  Step 3: If the result exceeds {format_val(class_threshold)}, classify as 'CRITICAL'. "
        f"Otherwise classify as 'STABLE'."
    )
    question = (
        f"Follow the multi-step procedure in the memo (using corrected data). "
        f"What is the final classification: CRITICAL or STABLE?"
    )
    rationale = (
        f"Step 1: Highest {attrs[ai_max]} -> {best_entity} ({format_val(best_val)}). "
        f"Step 2: Its {attrs[ai_second]} = {format_val(second_val)} x {1 + adj_pct/100} = {format_val(adjusted)}. "
        f"Step 3: {format_val(adjusted)} {'>' if label == 'CRITICAL' else '<='} {format_val(class_threshold)} -> {label}."
    )
    return question, label, trap, rule, rationale


# ═══════════════════════════════════════════════════════════════════
# MAIN GENERATION LOOP
# ═══════════════════════════════════════════════════════════════════

def generate_all(n_per_domain=20):
    """Generate all CMAT samples."""
    domain_names = list(DOMAINS.keys())
    all_samples = []
    sample_counter = 0

    for domain_name in domain_names:
        # 20 samples per domain: I1-I5 × C1-C4 = 20 combos
        combos = []
        for i_level in range(1, 6):
            for c_level in range(1, 5):
                combos.append((i_level, c_level))

        random.shuffle(combos)
        combos = combos[:n_per_domain]

        for integration, conflict in sorted(combos):
            sample_counter += 1
            sid = f"cmat_{sample_counter:04d}"
            print(f"  Generating {sid} -- {domain_name} I{integration}_C{conflict} ...", end="")

            sample = generate_sample(sid, domain_name, integration, conflict)
            all_samples.append(sample)
            print(" OK")

    return all_samples


def write_metadata(samples):
    out_path = SCRIPT_DIR / "metadata.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n[OK] Wrote {len(samples)} records to {out_path}")


def write_dataset_card():
    card = textwrap.dedent("""\
    # Cross-Modal Attention Triage (CMAT) Dataset v1.0

    ## Overview
    Benchmark for multimodal LLM **cross-modal attention**: can the model
    integrate information from BOTH a text passage AND an image to produce
    a correct answer?  Every sample is designed so that neither modality
    alone is sufficient.

    ## Domains
    | # | Domain | Theme |
    |---|---|---|
    | 1 | Space Mission Control | Sensor telemetry + protocol overrides |
    | 2 | Alchemy Lab | Reagent properties + recipe rules |
    | 3 | City Planning Board | District stats + regulation memos |
    | 4 | Sports Analytics | Player metrics + coach adjustments |
    | 5 | Archaeological Survey | Excavation data + field note corrections |

    ## Difficulty Axes
    **Integration Depth (I1-I5)** — how many cross-modal reasoning hops:
    - I1: single adjustment (text % applied to image value)
    - I2: formula (text formula applied to two image values)
    - I3: conditional (text if-then checked against image data)
    - I4: aggregation (text filter + sum across image values)
    - I5: multi-hop chain (find max → adjust → classify)

    **Conflict Level (C1-C4)** — how noisy/misleading the inputs are:
    - C1: no corruption, clean text
    - C2: 1 corrupted image value, text provides correction
    - C3: 2 corruptions + decoy panel on image + ignore instruction
    - C4: 3 corruptions + long filler text burying corrections

    ## Schema (metadata.jsonl)
    | Field | Type | Description |
    |---|---|---|
    | id | str | Sample ID (cmat_NNNN) |
    | image | str | Relative path to PNG |
    | text_passage | str | The memo/text input |
    | question | str | Question requiring both modalities |
    | correct_answer | str | Ground-truth answer |
    | image_only_trap | str | Wrong answer from image alone |
    | domain | str | Domain name |
    | integration_depth | int | 1-5 |
    | conflict_level | int | 1-4 |
    | difficulty_cell | str | e.g. I3_C2 |
    | rationale | str | Step-by-step reasoning chain |

    ## Statistics
    - Total samples: 100
    - Domains: 5 × 20 each
    - Difficulty cells: 20 (I1-I5 × C1-C4)
    - 100% synthetic, zero real-world data overlap

    ## Provenance
    Generated by generate_cmat_dataset.py — fully procedural, reproducible.
    """)
    card_path = SCRIPT_DIR / "dataset_card.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"[OK] Wrote dataset card to {card_path}")


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  CMAT Dataset Generator v1.0")
    print("  Cross-Modal Attention Triage -- 100 samples")
    print("=" * 60)
    print()

    samples = generate_all(n_per_domain=20)
    write_metadata(samples)
    write_dataset_card()

    # Summary
    print()
    print("=" * 60)
    print(f"  GENERATION COMPLETE")
    print(f"  Total samples : {len(samples)}")
    print(f"  Images dir    : {IMAGES_DIR}")
    print(f"  Metadata      : {SCRIPT_DIR / 'metadata.jsonl'}")
    print(f"  Dataset card  : {SCRIPT_DIR / 'dataset_card.md'}")
    print("=" * 60)

    # Distribution summary
    from collections import Counter
    domain_counts = Counter(s["domain"] for s in samples)
    diff_counts = Counter(s["difficulty_cell"] for s in samples)

    print("\n  Samples per domain:")
    for d, c in sorted(domain_counts.items()):
        print(f"    {d}: {c}")

    print("\n  Samples per difficulty cell:")
    for d, c in sorted(diff_counts.items()):
        print(f"    {d}: {c}")
