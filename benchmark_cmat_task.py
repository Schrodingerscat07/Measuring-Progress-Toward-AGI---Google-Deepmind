"""
CMAT v2.0 — Cross-Modal Attention Triage — FINAL BENCHMARK
============================================================
Designed for UNATTENDED execution on Kaggle servers.
- Auto-saves after every model (survives kernel restart)
- Auto-loads previous results on startup (add models tomorrow)
- Skips already-completed models (no wasted credits)
- All graphs + stats generated at the end

HOW TO USE:
1. Paste each CELL block into a separate Kaggle notebook cell
2. Click "Save Version" -> "Save & Run All (Commit)"
3. Close your laptop. Come back tomorrow and results will be ready.
4. To add more models later, just add new run_model() calls and re-run.
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 :  SETUP & DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

import os, json, base64, re, time, warnings
from collections import defaultdict
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

import kaggle_benchmarks as kbench
from kaggle_benchmarks.content_types import images
from kaggle_benchmarks.assertions import AssertionResult

os.environ["RENDER_SUBRUNS"] = "False"
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120, "font.size": 10})

# ── Locate dataset ────────────────────────────────────────────────────────────
def find_metadata():
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f == "metadata.jsonl":
                p = os.path.join(root, f)
                with open(p) as fh:
                    rec = json.loads(fh.readline())
                    if rec.get("id", "").startswith("cmat_"):
                        return p
    raise FileNotFoundError("CMAT metadata.jsonl not found in /kaggle/input")

META_PATH = find_metadata()
DATASET_ROOT = os.path.dirname(META_PATH)
print(f"[OK] Dataset root: {DATASET_ROOT}")

records = []
with open(META_PATH) as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))
df = pd.DataFrame(records)
df["image_path"] = df["image"].apply(lambda x: os.path.join(DATASET_ROOT, x))
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].fillna("")

print(f"[OK] Loaded {len(df)} samples | Domains: {df['domain'].nunique()} | Cells: {df['difficulty_cell'].nunique()}")

# ── Answer matching ───────────────────────────────────────────────────────────
def normalise(t):
    t = str(t).strip().lower()
    return re.sub(r"[,$\s]+", "", t)

def extract_number(t):
    m = re.search(r"(-?\d+\.?\d*)", normalise(t))
    return float(m.group(1)) if m else None

def answers_match(pred, correct):
    p, c = normalise(pred), normalise(correct)
    if c in p:
        return True
    if c in ("critical", "stable"):
        return c in p
    pn, cn = extract_number(pred), extract_number(correct)
    if pn is not None and cn is not None:
        if cn == 0:
            return abs(pn) < 0.5
        return abs(pn - cn) / abs(cn) <= 0.02
    return False

# ── Save / Load helpers ──────────────────────────────────────────────────────
BACKUP_FILE = "cmat_results_backup.json"

def save_progress():
    """Save ALL_RESULTS and ALL_SUMMARY to disk so they survive kernel restarts."""
    with open(BACKUP_FILE, "w") as f:
        json.dump({"summary": ALL_SUMMARY, "results": ALL_RESULTS}, f)
    print(f"  [SAVED] Progress backed up to {BACKUP_FILE}")

def load_progress():
    """Load previously saved results from disk."""
    if os.path.exists(BACKUP_FILE):
        with open(BACKUP_FILE, "r") as f:
            data = json.load(f)
        return data.get("summary", {}), data.get("results", {})
    return {}, {}

print("[OK] Cell 1 complete")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 :  KBENCH TASK DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# Global collector -- inner task appends here instead of returning dicts
SAMPLE_COLLECTOR = []

@kbench.assertions.assertion_handler()
def check_cross_modal(sample_id, correct_answer, image_only_trap, domain,
                      integration_depth, conflict_level, difficulty_cell,
                      model_response) -> AssertionResult:
    passed = answers_match(model_response, correct_answer)
    outcome = "correct" if passed else ("image_only_trap" if answers_match(model_response, image_only_trap) else "other_error")
    return AssertionResult(
        passed=passed,
        expectation=f"[{sample_id}] Exp='{correct_answer}' Got='{model_response.strip()[:80]}' -> {outcome}",
        details={"sample_id": sample_id, "response": model_response.strip()[:200],
                 "correct": correct_answer, "trap": image_only_trap, "outcome": outcome,
                 "domain": domain, "I": integration_depth, "C": conflict_level, "cell": difficulty_cell},
    )


@kbench.task(store_task=False)
def single_cmat_task(llm, image_path, text_passage, question, correct_answer,
                     image_only_trap, id, domain, integration_depth,
                     conflict_level, difficulty_cell, **kw):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    img = images.from_base64(b64, format="png")

    prompt = (
        f"You are given a data dashboard image AND a text memo. "
        f"Read BOTH carefully.\n\n"
        f"--- TEXT MEMO ---\n{text_passage}\n\n"
        f"--- QUESTION ---\n{question}\n\n"
        f"Reply with ONLY the final answer (a number or CRITICAL/STABLE). No explanation."
    )
    with kbench.chats.new(f"cmat_{id}"):
        resp = llm.prompt(prompt, image=img)
    model_response = resp.text if hasattr(resp, "text") else str(resp)

    check_cross_modal(sample_id=id, correct_answer=correct_answer,
                      image_only_trap=image_only_trap, domain=domain,
                      integration_depth=integration_depth,
                      conflict_level=conflict_level,
                      difficulty_cell=difficulty_cell,
                      model_response=model_response)

    is_correct = answers_match(model_response, correct_answer)
    hit_trap = (not is_correct) and answers_match(model_response, image_only_trap)

    # Store in global collector (kbench inner tasks must NOT return dicts)
    SAMPLE_COLLECTOR.append({
        "id": id, "correct_answer": correct_answer,
        "predicted": model_response.strip()[:200],
        "is_correct": is_correct, "hit_image_trap": hit_trap,
        "domain": domain, "integration_depth": integration_depth,
        "conflict_level": conflict_level, "difficulty_cell": difficulty_cell,
    })


@kbench.task(
    name="CMAT-v2.0",
    description=(
        "Cross-Modal Attention Triage v2.0: 500 samples across 10 domains. "
        "Tests whether models can integrate text memos + images to perform "
        "corrections, arithmetic, conditionals, aggregation, and multi-hop reasoning."
    ),
)
def cmat_benchmark(llm):
    # Clear collector before each model run
    SAMPLE_COLLECTOR.clear()

    with kbench.client.enable_cache():
        single_cmat_task.evaluate(
            stop_condition=lambda r: len(r) == df.shape[0],
            max_attempts=1, llm=[llm], evaluation_data=df,
            n_jobs=4, timeout=180, remove_run_files=True,
        )

    # Read results from global collector
    results = pd.DataFrame(SAMPLE_COLLECTOR)
    total = len(results)
    correct = int(results["is_correct"].sum()) if total > 0 else 0
    traps = int(results["hit_image_trap"].sum()) if total > 0 else 0
    acc = correct / total if total > 0 else 0.0
    std = float(results["is_correct"].std()) if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  CMAT v2.0 RESULTS  |  Acc: {acc:.1%} +/- {std:.3f}  |  {correct}/{total}")
    print(f"  Traps: {traps}  |  Other errors: {total - correct - traps}")
    if total > 0:
        for il in sorted(results["integration_depth"].unique()):
            sub = results[results["integration_depth"] == il]
            a = sub["is_correct"].mean()
            print(f"  I{il}: {a:.0%} ({int(sub['is_correct'].sum())}/{len(sub)})", end="  ")
    print(f"\n{'='*60}\n")

    kbench.assertions.assert_true(acc >= 0,
        expectation=f"Accuracy: {acc:.4f} +/- {std:.4f} on {total} samples")
    return None

print("[OK] Cell 2 complete - task definitions ready")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3a :  MODEL CONFIG & HELPER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

# Model metadata for analysis & graphs
MODEL_INFO = {
    # -- Google Gemma (Open-Source, Vision) --
    "google/gemma-3-4b":             {"short":"Gemma3-4B",    "provider":"Google","family":"Gemma", "gen":3.0, "params":"4B",   "tier":"open-small",   "order":1},
    "google/gemma-3-12b":            {"short":"Gemma3-12B",   "provider":"Google","family":"Gemma", "gen":3.0, "params":"12B",  "tier":"open-mid",     "order":2},
    "google/gemma-3-27b":            {"short":"Gemma3-27B",   "provider":"Google","family":"Gemma", "gen":3.0, "params":"27B",  "tier":"open-mid",     "order":3},
    # "google/gemma-4-26b-a4b":        -- EXCLUDED (only 15/500 samples completed in overnight run)
    "google/gemma-4-31b":            {"short":"Gemma4-31B",   "provider":"Google","family":"Gemma", "gen":4.0, "params":"31B",  "tier":"open-mid",     "order":5},
    # -- Google Gemini (Frontier) --
    "google/gemini-2.0-flash-lite":  {"short":"Gem2.0-FL",    "provider":"Google","family":"Gemini","gen":2.0, "params":"?",    "tier":"frontier-lite", "order":6},
    "google/gemini-2.0-flash":       {"short":"Gem2.0-F",     "provider":"Google","family":"Gemini","gen":2.0, "params":"?",    "tier":"frontier-mid",  "order":7},
    "google/gemini-2.5-flash":       {"short":"Gem2.5-F",     "provider":"Google","family":"Gemini","gen":2.5, "params":"?",    "tier":"frontier-mid",  "order":8},
    "google/gemini-2.5-pro":         {"short":"Gem2.5-P",     "provider":"Google","family":"Gemini","gen":2.5, "params":"?",    "tier":"frontier-top",  "order":9},
    "google/gemini-3-flash-preview": {"short":"Gem3.0-F",     "provider":"Google","family":"Gemini","gen":3.0, "params":"?",    "tier":"frontier-mid",  "order":10},
    "google/gemini-3.1-flash-lite-preview":{"short":"Gem3.1-FL","provider":"Google","family":"Gemini","gen":3.1,"params":"?",   "tier":"frontier-lite", "order":11},
    "google/gemini-3.1-pro-preview": {"short":"Gem3.1-P",     "provider":"Google","family":"Gemini","gen":3.1, "params":"?",    "tier":"frontier-top",  "order":12},
    # -- OpenAI --
    "openai/gpt-oss-20b":            {"short":"GPT-OSS-20B",  "provider":"OpenAI","family":"GPT-OSS","gen":5.0,"params":"20B",  "tier":"open-small",   "order":13},
    # "openai/gpt-oss-120b":           -- EXCLUDED (only 9/500 samples, 0% accuracy, model failed)
    "openai/gpt-5.4-nano-2026-03-17":{"short":"GPT5.4-Nano",  "provider":"OpenAI","family":"GPT",   "gen":5.4, "params":"?",   "tier":"frontier-lite", "order":15},
    "openai/gpt-5.4-mini-2026-03-17":{"short":"GPT5.4-Mini",  "provider":"OpenAI","family":"GPT",   "gen":5.4, "params":"?",   "tier":"frontier-mid",  "order":16},
    "openai/gpt-5.4-2026-03-05":     {"short":"GPT5.4",       "provider":"OpenAI","family":"GPT",   "gen":5.4, "params":"?",   "tier":"frontier-top",  "order":17},
    # -- Anthropic --
    "anthropic/claude-haiku-4-5@20251001":{"short":"Haiku4.5", "provider":"Anthropic","family":"Claude","gen":4.5,"params":"?", "tier":"frontier-lite", "order":18},
    "anthropic/claude-sonnet-4@20250514": {"short":"Sonnet4",  "provider":"Anthropic","family":"Claude","gen":4.0,"params":"?", "tier":"frontier-mid",  "order":19},
    # -- Others --
    "qwen/qwen3-235b-a22b-instruct-2507":{"short":"Qwen3-235B","provider":"Qwen","family":"Qwen","gen":3.0,"params":"235B",   "tier":"open-large",   "order":20},
    "qwen/qwen3-next-80b-a3b-instruct":  {"short":"Qwen3N-80B","provider":"Qwen","family":"Qwen","gen":3.5,"params":"80B",    "tier":"open-mid",     "order":21},
    "zai/glm-5":                     {"short":"GLM-5",        "provider":"ZAI",  "family":"GLM",  "gen":5.0, "params":"?",     "tier":"frontier-mid",  "order":22},
}

# ── Load any previously saved results ─────────────────────────────────────────
ALL_SUMMARY, ALL_RESULTS = load_progress()
if ALL_SUMMARY:
    print(f"[RESTORED] Loaded {len([m for m,s in ALL_SUMMARY.items() if s.get('accuracy',-1)>=0])} previously completed models from backup!")
    for m, s in ALL_SUMMARY.items():
        if s.get("accuracy", -1) >= 0:
            print(f"  - {s.get('short', m)}: {s['accuracy']:.1%}")
else:
    print("[INFO] No previous backup found - starting fresh")

test_img = df.iloc[0]["image_path"]


def run_model(model_name):
    """
    Run the full 500-sample CMAT benchmark on a single model.
    - SKIPS if model was already completed (loaded from backup)
    - AUTO-SAVES results to disk after completion
    - Results accumulate in ALL_RESULTS & ALL_SUMMARY across cells

    Usage:
        run_model("google/gemma-3-4b")
        run_model("google/gemini-2.5-flash")
    """
    # Skip if already completed
    if model_name in ALL_SUMMARY and ALL_SUMMARY[model_name].get("accuracy", -1) >= 0:
        prev = ALL_SUMMARY[model_name]
        print(f"\n  [SKIP] {model_name} already completed: {prev['accuracy']:.1%} ({prev.get('n_correct',0)}/{prev.get('n_total',0)})")
        return

    # Auto-register unknown models
    if model_name not in MODEL_INFO:
        MODEL_INFO[model_name] = {
            "short": model_name.split("/")[-1][:16],
            "provider": model_name.split("/")[0],
            "family": "?", "gen": 0, "params": "?",
            "tier": "unknown", "order": 99,
        }
    info = MODEL_INFO[model_name]

    print(f"\n{'='*70}")
    print(f"  MODEL: {model_name}  ({info['short']})")
    print(f"{'='*70}")

    # Get model handle
    try:
        model = kbench.llms[model_name]
    except Exception as e:
        print(f"  -> SKIP: model not available ({e})")
        return

    # Vision test
    print(f"  -> Testing vision capability...", end=" ", flush=True)
    try:
        with open(test_img, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        img = images.from_base64(b64, format="png")
        with kbench.chats.new("vision_test"):
            model.prompt("Reply YES", image=img)
        print("OK")
    except Exception as e:
        emsg = str(e).lower()
        if "image" in emsg or "vision" in emsg or "modality" in emsg:
            print(f"NO VISION -- skipping")
            return
        print(f"OK (non-vision error ignored)")

    # Run full evaluation
    t0 = time.time()
    try:
        cmat_benchmark.run(model)
        elapsed = time.time() - t0

        saved = list(SAMPLE_COLLECTOR)
        ALL_RESULTS[model_name] = saved

        res_df = pd.DataFrame(saved)
        n = len(res_df)
        acc = float(res_df["is_correct"].mean()) if n > 0 else 0.0
        std = float(res_df["is_correct"].std()) if n > 0 else 0.0
        n_correct = int(res_df["is_correct"].sum()) if n > 0 else 0

        ALL_SUMMARY[model_name] = {
            "accuracy": acc, "std": std, "time_s": elapsed,
            "n_correct": n_correct, "n_total": n, **info,
        }
        print(f"  -> DONE: {acc:.1%} ({n_correct}/{n}) in {elapsed:.0f}s")

        # AUTO-SAVE after every successful model
        save_progress()

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  -> FAILED after {elapsed:.0f}s: {e}")
        # Save partial results if any were collected
        saved = list(SAMPLE_COLLECTOR)
        if len(saved) > 0:
            ALL_RESULTS[model_name] = saved
            res_df = pd.DataFrame(saved)
            n = len(res_df)
            acc = float(res_df["is_correct"].mean()) if n > 0 else 0.0
            std = float(res_df["is_correct"].std()) if n > 0 else 0.0
            n_correct = int(res_df["is_correct"].sum()) if n > 0 else 0
            ALL_SUMMARY[model_name] = {
                "accuracy": acc, "std": std, "time_s": elapsed,
                "n_correct": n_correct, "n_total": n, **info,
                "partial": True,
            }
            print(f"  -> PARTIAL SAVE: {acc:.1%} ({n_correct}/{n} of {len(df)}) samples collected")
            save_progress()
        else:
            ALL_SUMMARY[model_name] = {
                "accuracy": -1, "std": 0, "time_s": elapsed,
                "n_correct": 0, "n_total": 0, **info,
            }

    # Print running totals
    done = [m for m, s in ALL_SUMMARY.items() if s.get("accuracy", -1) >= 0]
    print(f"\n  Models completed so far: {len(done)}")
    for m in done:
        s = ALL_SUMMARY[m]
        partial = " (PARTIAL)" if s.get("partial") else ""
        print(f"    {s.get('short', m):<20} {s['accuracy']:.1%}  ({s.get('n_correct',0)}/{s.get('n_total',0)}){partial}")

print("[OK] run_model() helper ready")
print(f"[OK] Dataset: {len(df)} samples")
print(f"\nUsage:  run_model('google/gemma-3-4b')")
print(f"\nModels will auto-skip if already completed. Results auto-save after each model.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3b :  Google Gemma Models (open-source)
# ══════════════════════════════════════════════════════════════════════════════
run_model("google/gemma-3-4b")
run_model("google/gemma-3-12b")
run_model("google/gemma-3-27b")
# run_model("google/gemma-4-26b-a4b")   # EXCLUDED - only 15/500 samples completed
run_model("google/gemma-4-31b")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3c :  Google Gemini Models (frontier)
# ══════════════════════════════════════════════════════════════════════════════
run_model("google/gemini-2.0-flash-lite")
run_model("google/gemini-2.0-flash")
run_model("google/gemini-2.5-flash")
run_model("google/gemini-2.5-pro")
run_model("google/gemini-3-flash-preview")
run_model("google/gemini-3.1-flash-lite-preview")
run_model("google/gemini-3.1-pro-preview")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3d :  OpenAI Models
# ══════════════════════════════════════════════════════════════════════════════
run_model("openai/gpt-oss-20b")
# run_model("openai/gpt-oss-120b")      # EXCLUDED - only 9/500 samples, model failed
run_model("openai/gpt-5.4-nano-2026-03-17")
run_model("openai/gpt-5.4-mini-2026-03-17")
run_model("openai/gpt-5.4-2026-03-05")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3e :  Other Models (Qwen, GLM)
# ══════════════════════════════════════════════════════════════════════════════
run_model("qwen/qwen3-235b-a22b-instruct-2507")
run_model("qwen/qwen3-next-80b-a3b-instruct")
run_model("zai/glm-5")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3f :  Anthropic Models
# ══════════════════════════════════════════════════════════════════════════════
run_model("anthropic/claude-haiku-4-5@20251001")
run_model("anthropic/claude-sonnet-4@20250514")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3g :  FINAL BACKUP & STATUS CHECK
# ══════════════════════════════════════════════════════════════════════════════
save_progress()

done = [m for m, s in ALL_SUMMARY.items() if s.get("accuracy", -1) >= 0]
failed = [m for m, s in ALL_SUMMARY.items() if s.get("accuracy", -1) < 0]

print(f"\n{'='*70}")
print(f"  EVALUATION COMPLETE")
print(f"  Successfully tested: {len(done)} models")
print(f"  Failed: {len(failed)} models")
print(f"{'='*70}")

if done:
    print(f"\n  COMPLETED MODELS:")
    for m in done:
        s = ALL_SUMMARY[m]
        partial = " (PARTIAL)" if s.get("partial") else ""
        print(f"    {s.get('short', m):<20} {s['accuracy']:.1%}  ({s.get('n_correct',0)}/{s.get('n_total',0)})  {s.get('time_s',0):.0f}s{partial}")

if failed:
    print(f"\n  FAILED MODELS (can retry later):")
    for m in failed:
        print(f"    - {m}")

print(f"\n  Backup file: {BACKUP_FILE}")
print(f"  To add more models tomorrow, just add run_model() calls and re-run.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 :  COMPREHENSIVE ANALYSIS & GRAPHS
# ══════════════════════════════════════════════════════════════════════════════

# If ALL_SUMMARY is empty (kernel restarted), reload from backup
if not ALL_SUMMARY:
    ALL_SUMMARY, ALL_RESULTS = load_progress()
    print(f"[RESTORED] Loaded {len(ALL_SUMMARY)} models from backup")

# ── 1. Build master results DataFrame ─────────────────────────────────────────
# Filter: exclude models that didn't complete enough samples
MIN_SAMPLES = 400  # Models with fewer samples are excluded from analysis

model_rows = []
for mname, summary in ALL_SUMMARY.items():
    if summary.get("accuracy", -1) >= 0:
        n_total = summary.get("n_total", 0)
        if n_total < MIN_SAMPLES:
            print(f"  [EXCLUDED] {summary.get('short', mname)}: only {n_total}/{len(df)} samples (minimum: {MIN_SAMPLES})")
            continue
        # Make sure MODEL_INFO has this model
        if mname not in MODEL_INFO:
            MODEL_INFO[mname] = {
                "short": summary.get("short", mname.split("/")[-1][:16]),
                "provider": summary.get("provider", mname.split("/")[0]),
                "family": summary.get("family", "?"),
                "gen": summary.get("gen", 0),
                "params": summary.get("params", "?"),
                "tier": summary.get("tier", "unknown"),
                "order": summary.get("order", 99),
            }
        model_rows.append({"model": mname, **summary})

if not model_rows:
    print("[ERROR] No completed models found. Run model cells first!")
else:
    mdf = pd.DataFrame(model_rows).sort_values("accuracy", ascending=False).reset_index(drop=True)

    # Build per-sample results DataFrame
    sample_rows = []
    for mname, results_list in ALL_RESULTS.items():
        summary = ALL_SUMMARY.get(mname, {})
        if summary.get("accuracy", -1) < 0:
            continue
        if summary.get("n_total", 0) < MIN_SAMPLES:
            continue
        for r in results_list:
            if isinstance(r, dict):
                sample_rows.append({"model": mname, **r})
    sdf = pd.DataFrame(sample_rows) if sample_rows else pd.DataFrame()

    PROVIDER_COLORS = {"Google": "#4285F4", "OpenAI": "#10A37F", "Anthropic": "#D97706",
                       "Qwen": "#EF4444", "ZAI": "#8B5CF6", "?": "#6B7280"}


    # ── 2. Print Leaderboard ──────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  CMAT v2.0 MODEL LEADERBOARD  ({len(df)} samples)")
    print(f"{'='*75}")
    print(f"  {'Rank':<5} {'Model':<35} {'Acc':>8} {'Time':>8}  Provider")
    print(f"  {'-'*70}")
    for i, row in mdf.iterrows():
        print(f"  {i+1:<5} {row['short']:<35} {row['accuracy']:>7.1%} {row['time_s']:>7.0f}s  {row.get('provider','?')}")
    print(f"{'='*75}\n")


    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 1: Overall Accuracy Leaderboard (horizontal bar)
    # ──────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, max(6, len(mdf) * 0.45)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    # Reverse mdf so lowest accuracy is at bottom (position 0), highest at top
    plot_df = mdf.iloc[::-1].reset_index(drop=True)
    colors = [PROVIDER_COLORS.get(r.get("provider", "?"), "#6B7280") for _, r in plot_df.iterrows()]
    bars = ax.barh(range(len(plot_df)), plot_df["accuracy"] * 100, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["short"].tolist(), color="white", fontsize=9)
    ax.set_xlabel("Accuracy (%)", color="white")
    ax.set_title("CMAT v2.0 -- Model Leaderboard", color="white", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 105)
    for i, (_, r) in enumerate(plot_df.iterrows()):
        ax.text(r["accuracy"] * 100 + 1, i, f'{r["accuracy"]:.1%}', va="center", color="white", fontsize=8)
    handles = [Patch(facecolor=c, label=p) for p, c in PROVIDER_COLORS.items() if p in mdf["provider"].values]
    ax.legend(handles=handles, loc="lower right", fontsize=8, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("graph_leaderboard.png", facecolor="#0d1117")
    plt.show()


    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 2: Accuracy by Integration Depth (line per model)
    # ──────────────────────────────────────────────────────────────────────────
    if len(sdf) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
        for mname in mdf["model"].tolist()[:10]:
            info = MODEL_INFO.get(mname, {})
            sub = sdf[sdf["model"] == mname]
            accs = sub.groupby("integration_depth")["is_correct"].mean()
            color = PROVIDER_COLORS.get(info.get("provider", "?"), "#aaa")
            ax.plot(accs.index, accs.values * 100, "o-", label=info.get("short", mname[:15]),
                    color=color, alpha=0.85, linewidth=2, markersize=5)
        ax.set_xticks([1,2,3,4,5])
        ax.set_xticklabels(["I1\nSimple %", "I2\nFormula", "I3\nConditional", "I4\nAggregate", "I5\nMulti-hop"], color="white", fontsize=9)
        ax.set_ylabel("Accuracy (%)", color="white")
        ax.set_title("Accuracy by Integration Depth", color="white", fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, loc="lower left", facecolor="#1a1a2e", edgecolor="white", labelcolor="white", ncol=2)
        ax.set_ylim(-5, 105)
        ax.tick_params(colors="white")
        ax.grid(axis="y", alpha=0.15, color="white")
        for s in ax.spines.values(): s.set_color("#333")
        plt.tight_layout(); plt.savefig("graph_integration_depth.png", facecolor="#0d1117"); plt.show()


    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 3: Accuracy by Conflict Level (line per model)
    # ──────────────────────────────────────────────────────────────────────────
    if len(sdf) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
        for mname in mdf["model"].tolist()[:10]:
            info = MODEL_INFO.get(mname, {})
            sub = sdf[sdf["model"] == mname]
            accs = sub.groupby("conflict_level")["is_correct"].mean()
            color = PROVIDER_COLORS.get(info.get("provider", "?"), "#aaa")
            ax.plot(accs.index, accs.values * 100, "s-", label=info.get("short", mname[:15]),
                    color=color, alpha=0.85, linewidth=2, markersize=5)
        ax.set_xticks([1,2,3,4,5])
        ax.set_xticklabels(["C1\nClean", "C2\n1 corrupt", "C3\n2+decoy", "C4\n3+filler", "C5\nAdversarial"], color="white", fontsize=9)
        ax.set_ylabel("Accuracy (%)", color="white")
        ax.set_title("Conflict Resilience", color="white", fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, loc="lower left", facecolor="#1a1a2e", edgecolor="white", labelcolor="white", ncol=2)
        ax.set_ylim(-5, 105)
        ax.tick_params(colors="white")
        ax.grid(axis="y", alpha=0.15, color="white")
        for s in ax.spines.values(): s.set_color("#333")
        plt.tight_layout(); plt.savefig("graph_conflict_level.png", facecolor="#0d1117"); plt.show()


    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 4: Difficulty Cell Heatmap (I x C, averaged across all models)
    # ──────────────────────────────────────────────────────────────────────────
    if len(sdf) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
        pivot = sdf.pivot_table(values="is_correct", index="conflict_level",
                                columns="integration_depth", aggfunc="mean") * 100
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(5)); ax.set_xticklabels([f"I{i}" for i in range(1,6)], color="white")
        ax.set_yticks(range(5)); ax.set_yticklabels([f"C{i}" for i in range(1,6)], color="white")
        for i in range(5):
            for j in range(5):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        color="black" if val > 50 else "white", fontweight="bold", fontsize=11)
        ax.set_xlabel("Integration Depth", color="white")
        ax.set_ylabel("Conflict Level", color="white")
        ax.set_title("Difficulty Cell Heatmap (avg all models)", color="white", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Accuracy %")
        plt.tight_layout(); plt.savefig("graph_heatmap.png", facecolor="#0d1117"); plt.show()


    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 5: Domain Difficulty Ranking
    # ──────────────────────────────────────────────────────────────────────────
    if len(sdf) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
        dom_acc = sdf.groupby("domain")["is_correct"].mean().sort_values(ascending=False)
        bars = ax.barh(range(len(dom_acc)), dom_acc.values * 100,
                       color=plt.cm.viridis(np.linspace(0.2, 0.9, len(dom_acc))), edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(dom_acc)))
        ax.set_yticklabels([d.replace("_", " ").title() for d in dom_acc.index], color="white", fontsize=9)
        for i, v in enumerate(dom_acc.values):
            ax.text(v * 100 + 1, i, f"{v:.1%}", va="center", color="white", fontsize=9)
        ax.set_xlabel("Accuracy (%)", color="white")
        ax.set_title("Domain Difficulty Ranking (hardest at top)", color="white", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 105)
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#333")
        plt.tight_layout(); plt.savefig("graph_domain_ranking.png", facecolor="#0d1117"); plt.show()


    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 6: Failure Mode Breakdown (stacked bar)
    # ──────────────────────────────────────────────────────────────────────────
    if len(sdf) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
        models_ordered = mdf["model"].tolist()
        shorts = [MODEL_INFO.get(m, {}).get("short", m[:15]) for m in models_ordered]
        correct_pcts, trap_pcts, error_pcts = [], [], []
        for m in models_ordered:
            sub = sdf[sdf["model"] == m]
            n = len(sub) if len(sub) > 0 else 1
            correct_pcts.append(sub["is_correct"].sum() / n * 100)
            trap_pcts.append(sub["hit_image_trap"].sum() / n * 100)
            error_pcts.append((n - sub["is_correct"].sum() - sub["hit_image_trap"].sum()) / n * 100)
        x = range(len(models_ordered))
        ax.bar(x, correct_pcts, label="Correct", color="#22c55e", edgecolor="white", linewidth=0.3)
        ax.bar(x, trap_pcts, bottom=correct_pcts, label="Image Trap", color="#f59e0b", edgecolor="white", linewidth=0.3)
        bottoms = [c + t for c, t in zip(correct_pcts, trap_pcts)]
        ax.bar(x, error_pcts, bottom=bottoms, label="Other Error", color="#ef4444", edgecolor="white", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(shorts, rotation=45, ha="right", color="white", fontsize=8)
        ax.set_ylabel("% of Samples", color="white")
        ax.set_title("Failure Mode Breakdown", color="white", fontsize=13, fontweight="bold")
        ax.legend(facecolor="#1a1a2e", edgecolor="white", labelcolor="white", fontsize=9)
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#333")
        plt.tight_layout(); plt.savefig("graph_failure_modes.png", facecolor="#0d1117"); plt.show()


    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 7: Generational Scaling (scatter: model generation vs accuracy)
    # ──────────────────────────────────────────────────────────────────────────
    if len(mdf) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
        for _, r in mdf.iterrows():
            gen = r.get("gen", 0)
            if gen <= 0:
                continue
            color = PROVIDER_COLORS.get(r.get("provider", "?"), "#aaa")
            ax.scatter(gen, r["accuracy"] * 100, s=120, color=color, edgecolors="white",
                       linewidth=0.5, zorder=5)
            ax.annotate(r["short"], (gen, r["accuracy"] * 100),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=7, color="white", alpha=0.9)
        ax.set_xlabel("Model Generation", color="white", fontsize=11)
        ax.set_ylabel("Accuracy (%)", color="white", fontsize=11)
        ax.set_title("Generational Scaling -- Older to Newer", color="white", fontsize=13, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.15, color="white")
        handles = [Patch(facecolor=c, label=p) for p, c in PROVIDER_COLORS.items() if p in mdf["provider"].values]
        ax.legend(handles=handles, fontsize=8, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#333")
        plt.tight_layout(); plt.savefig("graph_generational.png", facecolor="#0d1117"); plt.show()


    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 8: Provider Head-to-Head (best model per provider)
    # ──────────────────────────────────────────────────────────────────────────
    if len(mdf) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
        best_per_provider = mdf.groupby("provider").apply(lambda g: g.loc[g["accuracy"].idxmax()]).reset_index(drop=True)
        best_per_provider = best_per_provider.sort_values("accuracy", ascending=True)
        colors = [PROVIDER_COLORS.get(p, "#aaa") for p in best_per_provider["provider"]]
        ax.barh(range(len(best_per_provider)), best_per_provider["accuracy"] * 100,
                color=colors, edgecolor="white", linewidth=0.5)
        labels = [f'{r["provider"]}\n({r["short"]})' for _, r in best_per_provider.iterrows()]
        ax.set_yticks(range(len(best_per_provider)))
        ax.set_yticklabels(labels, color="white", fontsize=9)
        for i, r in enumerate(best_per_provider.itertuples()):
            ax.text(r.accuracy * 100 + 1, i, f"{r.accuracy:.1%}", va="center", color="white", fontsize=9)
        ax.set_xlabel("Best Model Accuracy (%)", color="white")
        ax.set_title("Provider Head-to-Head (Best Model Each)", color="white", fontsize=13, fontweight="bold")
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#333")
        ax.set_xlim(0, 105)
        plt.tight_layout(); plt.savefig("graph_providers.png", facecolor="#0d1117"); plt.show()


    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 9: Arithmetic Competence (I1 vs I2+ accuracy per model)
    # ──────────────────────────────────────────────────────────────────────────
    if len(sdf) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
        models_list = mdf["model"].tolist()
        shorts = [MODEL_INFO.get(m, {}).get("short", m[:15]) for m in models_list]
        x = np.arange(len(models_list))
        w = 0.35
        i1_accs, i2plus_accs = [], []
        for m in models_list:
            sub = sdf[sdf["model"] == m]
            i1 = sub[sub["integration_depth"] == 1]["is_correct"].mean() * 100 if len(sub) > 0 else 0
            i2p = sub[sub["integration_depth"] >= 2]["is_correct"].mean() * 100 if len(sub) > 0 else 0
            i1_accs.append(i1)
            i2plus_accs.append(i2p)
        ax.bar(x - w/2, i1_accs, w, label="I1 (No Arithmetic)", color="#60a5fa", edgecolor="white", linewidth=0.3)
        ax.bar(x + w/2, i2plus_accs, w, label="I2-I5 (Requires Computation)", color="#f87171", edgecolor="white", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(shorts, rotation=45, ha="right", color="white", fontsize=8)
        ax.set_ylabel("Accuracy (%)", color="white")
        ax.set_title("Arithmetic Competence: Simple Lookup vs Computation", color="white", fontsize=13, fontweight="bold")
        ax.legend(facecolor="#1a1a2e", edgecolor="white", labelcolor="white", fontsize=9)
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#333")
        plt.tight_layout(); plt.savefig("graph_arithmetic.png", facecolor="#0d1117"); plt.show()


    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 10: Speed vs Accuracy Trade-off
    # ──────────────────────────────────────────────────────────────────────────
    if len(mdf) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
        for _, r in mdf.iterrows():
            color = PROVIDER_COLORS.get(r.get("provider", "?"), "#aaa")
            ax.scatter(r["time_s"], r["accuracy"] * 100, s=120, color=color,
                       edgecolors="white", linewidth=0.5, zorder=5)
            ax.annotate(r["short"], (r["time_s"], r["accuracy"] * 100),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=7, color="white", alpha=0.9)
        ax.set_xlabel("Total Evaluation Time (seconds)", color="white")
        ax.set_ylabel("Accuracy (%)", color="white")
        ax.set_title("Speed vs Accuracy Trade-off", color="white", fontsize=13, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.15, color="white")
        handles = [Patch(facecolor=c, label=p) for p, c in PROVIDER_COLORS.items() if p in mdf["provider"].values]
        ax.legend(handles=handles, fontsize=8, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#333")
        plt.tight_layout(); plt.savefig("graph_speed_accuracy.png", facecolor="#0d1117"); plt.show()


    # ──────────────────────────────────────────────────────────────────────────
    # DETAILED STATISTICS TABLE
    # ──────────────────────────────────────────────────────────────────────────
    if len(sdf) > 0:
        print(f"\n{'='*90}")
        print(f"  DETAILED STATISTICS")
        print(f"{'='*90}")

        # Per-model breakdown by I level
        print(f"\n  Accuracy by Model x Integration Depth:")
        print(f"  {'Model':<20} {'I1':>6} {'I2':>6} {'I3':>6} {'I4':>6} {'I5':>6} {'Overall':>8}")
        print(f"  {'-'*62}")
        for m in mdf["model"].tolist():
            info = MODEL_INFO.get(m, {})
            sub = sdf[sdf["model"] == m]
            cells = []
            for il in range(1, 6):
                a = sub[sub["integration_depth"] == il]["is_correct"].mean()
                cells.append(f"{a:.0%}")
            overall = sub["is_correct"].mean()
            print(f"  {info.get('short', m[:20]):<20} {cells[0]:>6} {cells[1]:>6} {cells[2]:>6} {cells[3]:>6} {cells[4]:>6} {overall:>7.1%}")

        # Per-model breakdown by C level
        print(f"\n  Accuracy by Model x Conflict Level:")
        print(f"  {'Model':<20} {'C1':>6} {'C2':>6} {'C3':>6} {'C4':>6} {'C5':>6} {'Overall':>8}")
        print(f"  {'-'*62}")
        for m in mdf["model"].tolist():
            info = MODEL_INFO.get(m, {})
            sub = sdf[sdf["model"] == m]
            cells = []
            for cl in range(1, 6):
                a = sub[sub["conflict_level"] == cl]["is_correct"].mean()
                cells.append(f"{a:.0%}")
            overall = sub["is_correct"].mean()
            print(f"  {info.get('short', m[:20]):<20} {cells[0]:>6} {cells[1]:>6} {cells[2]:>6} {cells[3]:>6} {cells[4]:>6} {overall:>7.1%}")

        # Image trap rate
        print(f"\n  Image-Only Trap Rate (higher = model ignores text more):")
        print(f"  {'Model':<20} {'Trap Rate':>10} {'Traps/Total':>12}")
        print(f"  {'-'*45}")
        for m in mdf["model"].tolist():
            info = MODEL_INFO.get(m, {})
            sub = sdf[sdf["model"] == m]
            n = len(sub) if len(sub) > 0 else 1
            traps = int(sub["hit_image_trap"].sum())
            print(f"  {info.get('short', m[:20]):<20} {traps/n:>9.1%} {traps:>5}/{n}")

        # Multi-hop degradation
        print(f"\n  Multi-Hop Degradation (I1 accuracy - I5 accuracy):")
        print(f"  {'Model':<20} {'I1':>6} {'I5':>6} {'Drop':>8}")
        print(f"  {'-'*42}")
        for m in mdf["model"].tolist():
            info = MODEL_INFO.get(m, {})
            sub = sdf[sdf["model"] == m]
            i1 = sub[sub["integration_depth"] == 1]["is_correct"].mean()
            i5 = sub[sub["integration_depth"] == 5]["is_correct"].mean()
            drop = i1 - i5
            print(f"  {info.get('short', m[:20]):<20} {i1:>5.0%} {i5:>5.0%} {drop:>7.0%}")

    print(f"\n\n{'='*70}")
    print(f"  ALL ANALYSIS COMPLETE -- graphs saved as PNG")
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 :  SELECT FOR LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════

# %choose cmat_benchmark
