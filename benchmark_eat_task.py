"""
EAT-Bench v1.0 — Emotional Attention Test — BENCHMARK EVALUATION
==================================================================
Designed for UNATTENDED execution on Kaggle servers.
- Auto-saves after every model (survives kernel restart)
- Auto-loads previous results on startup (add models tomorrow)
- Skips already-completed models (no wasted credits)
- All 10 analysis graphs + stats generated at the end

HOW TO USE:
1. Upload task_03_eat/ folder as a Kaggle dataset
2. Paste each CELL block into a separate Kaggle notebook cell
3. Click "Save Version" -> "Save & Run All (Commit)"
4. Close your laptop. Come back later and results will be ready.
5. To add more models later, just add new run_model() calls and re-run.
"""

# ==============================================================================
# CELL 1 :  SETUP & DATA LOADING
# ==============================================================================

import os, json, re, time, warnings
from collections import defaultdict
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    import matplotlib.ticker as mticker
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    import matplotlib.ticker as mticker

try:
    import seaborn as sns
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn", "-q"])
    import seaborn as sns

import kaggle_benchmarks as kbench
from kaggle_benchmarks.assertions import AssertionResult

os.environ["RENDER_SUBRUNS"] = "False"
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120, "font.size": 10,
                      "axes.titlesize": 12, "axes.labelsize": 10})

# -- Locate dataset --------------------------------------------------------
def find_eat_metadata():
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f == "metadata.jsonl":
                p = os.path.join(root, f)
                with open(p) as fh:
                    rec = json.loads(fh.readline())
                    if rec.get("id", "").startswith("eat_"):
                        return p
    raise FileNotFoundError("EAT metadata.jsonl not found in /kaggle/input")

META_PATH = find_eat_metadata()
DATASET_ROOT = os.path.dirname(META_PATH)
print(f"[OK] Dataset root: {DATASET_ROOT}")

records = []
with open(META_PATH) as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))
df = pd.DataFrame(records)
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].fillna("")

n_tasks = df["task_id"].nunique()
n_conds = df["condition"].nunique()
print(f"[OK] Loaded {len(df)} samples | Tasks: {n_tasks} | Conditions: {n_conds}")
print(f"     Task types: {sorted(df['task_type'].unique())}")
print(f"     Conditions: {sorted(df['condition'].unique())}")

# -- Robust answer matching ------------------------------------------------
def normalise(t):
    """Strip whitespace, lowercase, remove $, commas, %, degree symbols."""
    t = str(t).strip().lower()
    # Remove common formatting characters
    t = re.sub(r"[$,%°\s]+", "", t)
    # Remove trailing period
    t = t.rstrip(".")
    return t

def extract_number(t):
    """Extract the first number (including negative and decimal) from text."""
    clean = normalise(t)
    m = re.search(r"(-?\d+\.?\d*)", clean)
    return float(m.group(1)) if m else None

def extract_all_numbers(t):
    """Extract ALL numbers from text (for finding answer in chain-of-thought)."""
    clean = normalise(t)
    return [float(m) for m in re.findall(r"-?\d+\.?\d*", clean)]

def answers_match(predicted, correct):
    """
    Robust answer matching for EAT-Bench.
    Handles: case-insensitive text, $/%/degree stripping, numeric tolerance,
    and verbose model responses like "The answer is 42".

    Logic order:
      1. Exact match (normalised)
      2. Text answers → word-boundary match (Yes/No/True/False/Carol)
      3. Numeric answers → extract numbers and check tolerance
         Uses LAST number in response (models tend to conclude with the answer)
         Falls back to ANY number matching within tolerance
    """
    p = normalise(predicted)
    c = normalise(correct)

    # 1. Exact match after normalisation
    if c == p:
        return True

    # 2. Text answers (Yes/No/True/False/Carol) — use word-boundary, not substring
    if c in ("yes", "no", "true", "false", "carol"):
        # Word-boundary check on the RAW (pre-normalise) prediction for safety
        raw_lower = str(predicted).strip().lower()
        if re.search(r'\b' + re.escape(c) + r'\b', raw_lower):
            return True
        # Fallback: exact substring in normalised (handles edge cases)
        return c in p

    # 3. Numeric answers — proper extraction, NO naive substring
    cn = extract_number(correct)
    if cn is not None:
        all_nums = extract_all_numbers(predicted)
        if not all_nums:
            return False

        # Prefer the LAST number (models conclude with "the answer is X")
        last_num = all_nums[-1]
        if cn == 0:
            if abs(last_num) < 0.5:
                return True
        elif abs(last_num - cn) / abs(cn) <= 0.02:
            return True

        # Fallback: check if ANY number in the response matches
        for num in all_nums:
            if cn == 0:
                if abs(num) < 0.5:
                    return True
            elif abs(num - cn) / abs(cn) <= 0.02:
                return True

        return False

    return False


# -- Save / Load helpers ---------------------------------------------------
BACKUP_FILE = "eat_results_backup.json"

def save_progress():
    """Save ALL_RESULTS and ALL_SUMMARY to disk."""
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


# ==============================================================================
# CELL 2 :  KBENCH TASK DEFINITIONS
# ==============================================================================

# Global collector -- inner task appends here
SAMPLE_COLLECTOR = []

@kbench.assertions.assertion_handler()
def check_eat_answer(sample_id, correct_answer, condition, task_type,
                     difficulty, task_id, model_response) -> AssertionResult:
    passed = answers_match(model_response, correct_answer)
    return AssertionResult(
        passed=passed,
        expectation=f"[{sample_id}] Exp='{correct_answer}' Got='{model_response.strip()[:80]}' -> {'OK' if passed else 'WRONG'}",
        details={
            "sample_id": sample_id, "response": model_response.strip()[:200],
            "correct": correct_answer, "condition": condition,
            "task_type": task_type, "difficulty": difficulty, "task_id": task_id,
        },
    )


@kbench.task(store_task=False)
def single_eat_task(llm, prompt, correct_answer, id, condition, task_type,
                    difficulty, task_id, **kw):
    """Run a single EAT-Bench sample."""
    with kbench.chats.new(f"eat_{id}"):
        resp = llm.prompt(prompt)
    model_response = resp.text if hasattr(resp, "text") else str(resp)

    check_eat_answer(
        sample_id=id, correct_answer=correct_answer,
        condition=condition, task_type=task_type,
        difficulty=difficulty, task_id=task_id,
        model_response=model_response,
    )

    is_correct = answers_match(model_response, correct_answer)

    SAMPLE_COLLECTOR.append({
        "id": id, "task_id": task_id, "correct_answer": correct_answer,
        "predicted": model_response.strip()[:200],
        "is_correct": is_correct,
        "condition": condition, "task_type": task_type,
        "difficulty": difficulty,
    })


@kbench.task(
    name="EAT-Bench-v1.0",
    description=(
        "Emotional Attention Test v1.0: 300 samples (50 tasks x 6 conditions). "
        "Tests whether emotional framing (Fear, Urgency, Flattery, Grief, "
        "Existential Threat) systematically degrades AI reasoning accuracy "
        "compared to a neutral control."
    ),
)
def eat_benchmark(llm, eval_df):
    SAMPLE_COLLECTOR.clear()

    with kbench.client.enable_cache():
        single_eat_task.evaluate(
            stop_condition=lambda r: len(r) == eval_df.shape[0],
            max_attempts=1, llm=[llm], evaluation_data=eval_df,
            n_jobs=4, timeout=120, remove_run_files=True,
        )

    results = pd.DataFrame(SAMPLE_COLLECTOR)
    total = len(results)
    correct = int(results["is_correct"].sum()) if total > 0 else 0
    acc = correct / total if total > 0 else 0.0

    # Per-condition breakdown
    print(f"\n{'='*60}")
    print(f"  EAT-Bench v1.0 RESULTS  |  Overall: {acc:.1%}  ({correct}/{total})")
    print(f"{'='*60}")
    if total > 0:
        for cond in sorted(results["condition"].unique()):
            sub = results[results["condition"] == cond]
            a = sub["is_correct"].mean()
            print(f"  {cond:15s}: {a:.1%} ({int(sub['is_correct'].sum())}/{len(sub)})")
    print(f"{'='*60}\n")

    kbench.assertions.assert_true(acc >= 0,
        expectation=f"Overall accuracy: {acc:.4f} on {total} samples")
    return None

print("[OK] Cell 2 complete - task definitions ready")


# ==============================================================================
# CELL 3a :  MODEL CONFIG & HELPER FUNCTION
# ==============================================================================

MODEL_INFO = {
    # -- Google Gemma (Open-Source) --
    "google/gemma-3-1b":             {"short":"Gemma3-1B",    "provider":"Google","family":"Gemma", "gen":3.0, "params":"1B",   "tier":"open-tiny",    "order":1},
    "google/gemma-3-4b":             {"short":"Gemma3-4B",    "provider":"Google","family":"Gemma", "gen":3.0, "params":"4B",   "tier":"open-small",   "order":2},
    "google/gemma-3-27b":            {"short":"Gemma3-27B",   "provider":"Google","family":"Gemma", "gen":3.0, "params":"27B",  "tier":"open-mid",     "order":3},
    "google/gemma-4-31b":            {"short":"Gemma4-31B",   "provider":"Google","family":"Gemma", "gen":4.0, "params":"31B",  "tier":"open-mid",     "order":4},
    # -- Google Gemini (Frontier) --
    "google/gemini-2.0-flash-lite":  {"short":"Gem2.0-FL",    "provider":"Google","family":"Gemini","gen":2.0, "params":"?",    "tier":"frontier-lite", "order":5},
    "google/gemini-2.5-flash":       {"short":"Gem2.5-F",     "provider":"Google","family":"Gemini","gen":2.5, "params":"?",    "tier":"frontier-mid",  "order":6},
    "google/gemini-3.1-flash-lite-preview":{"short":"Gem3.1-FL","provider":"Google","family":"Gemini","gen":3.1,"params":"?",   "tier":"frontier-lite", "order":7},
    "google/gemini-3.1-pro-preview": {"short":"Gem3.1-P",     "provider":"Google","family":"Gemini","gen":3.1, "params":"?",    "tier":"frontier-top",  "order":8},
    # -- OpenAI --
    "openai/gpt-oss-20b":            {"short":"GPT-OSS-20B",  "provider":"OpenAI","family":"GPT-OSS","gen":5.0,"params":"20B",  "tier":"open-small",   "order":9},
    "openai/gpt-5.4-nano-2026-03-17":{"short":"GPT5.4-Nano",  "provider":"OpenAI","family":"GPT",   "gen":5.4, "params":"?",   "tier":"frontier-lite", "order":10},
    "openai/gpt-5.4-mini-2026-03-17":{"short":"GPT5.4-Mini",  "provider":"OpenAI","family":"GPT",   "gen":5.4, "params":"?",   "tier":"frontier-mid",  "order":11},
    "openai/gpt-5.4-2026-03-05":     {"short":"GPT5.4",       "provider":"OpenAI","family":"GPT",   "gen":5.4, "params":"?",   "tier":"frontier-top",  "order":12},
    # -- Anthropic --
    "anthropic/claude-haiku-4-5@20251001":{"short":"Haiku4.5", "provider":"Anthropic","family":"Claude","gen":4.5,"params":"?", "tier":"frontier-lite", "order":13},
    "anthropic/claude-sonnet-4@20250514": {"short":"Sonnet4",  "provider":"Anthropic","family":"Claude","gen":4.0,"params":"?", "tier":"frontier-mid",  "order":14},
    # -- DeepSeek (Reasoning) --
    "deepseek-ai/deepseek-r1-0528":  {"short":"DS-R1",        "provider":"DeepSeek","family":"DeepSeek","gen":1.0,"params":"671B","tier":"reasoning",  "order":15},
    # -- Qwen --
    "qwen/qwen3-235b-a22b-instruct-2507":{"short":"Qwen3-235B","provider":"Qwen","family":"Qwen","gen":3.0,"params":"235B",   "tier":"open-large",   "order":16},
    "qwen/qwen3-coder-480b-a35b-instruct":{"short":"Qwen3-Coder","provider":"Qwen","family":"Qwen",  "gen":3.0, "params":"480B",  "tier":"code-special", "order":17},
}

# Condition display names and colors for graphs
CONDITION_INFO = {
    "neutral":     {"display": "Neutral",     "color": "#6B7280"},  # Gray
    "fear":        {"display": "Fear",        "color": "#EF4444"},  # Red
    "urgency":     {"display": "Urgency",     "color": "#F59E0B"},  # Amber
    "flattery":    {"display": "Flattery",    "color": "#8B5CF6"},  # Purple
    "grief":       {"display": "Grief",       "color": "#3B82F6"},  # Blue
    "existential": {"display": "Existential", "color": "#10B981"},  # Green
}

PROVIDER_COLORS = {
    "Google": "#4285F4", "OpenAI": "#10A37F", "Anthropic": "#D97706",
    "DeepSeek": "#FF6B35", "Qwen": "#EF4444", "?": "#6B7280",
}

# -- Load previous results -------------------------------------------------
ALL_SUMMARY, ALL_RESULTS = load_progress()
if ALL_SUMMARY:
    done_count = len([m for m, s in ALL_SUMMARY.items() if s.get("accuracy", -1) >= 0])
    print(f"[RESTORED] Loaded {done_count} previously completed models from backup!")
    for m, s in ALL_SUMMARY.items():
        if s.get("accuracy", -1) >= 0:
            print(f"  - {s.get('short', m)}: {s['accuracy']:.1%}")
else:
    print("[INFO] No previous backup found - starting fresh")


def run_model(model_name):
    """
    Run the full 300-sample EAT-Bench on a single model.
    - SKIPS if model was already completed (loaded from backup)
    - AUTO-SAVES results to disk after completion
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

    # Quick connectivity test (text-only, no vision needed)
    print(f"  -> Testing connectivity...", end=" ", flush=True)
    try:
        with kbench.chats.new("test"):
            model.prompt("Reply with only the number 7")
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        return

    # Run full evaluation
    t0 = time.time()
    try:
        eat_benchmark.run(model, df)
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
        save_progress()

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  -> FAILED after {elapsed:.0f}s: {e}")
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
            print(f"  -> PARTIAL SAVE: {acc:.1%} ({n_correct}/{n} of {len(df)}) samples")
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


# ==============================================================================
# CELL 3b :  Google Gemma Models (open-source, text-only OK)
# ==============================================================================
run_model("google/gemma-3-1b")
run_model("google/gemma-3-4b")
run_model("google/gemma-3-27b")
run_model("google/gemma-4-31b")


# ==============================================================================
# CELL 3c :  Google Gemini Models (frontier)
# ==============================================================================
run_model("google/gemini-2.0-flash-lite")
run_model("google/gemini-2.5-flash")
run_model("google/gemini-3.1-flash-lite-preview")
run_model("google/gemini-3.1-pro-preview")


# ==============================================================================
# CELL 3d :  OpenAI Models
# ==============================================================================
run_model("openai/gpt-oss-20b")
run_model("openai/gpt-5.4-nano-2026-03-17")
run_model("openai/gpt-5.4-mini-2026-03-17")
run_model("openai/gpt-5.4-2026-03-05")


# ==============================================================================
# CELL 3e :  Other Models (Anthropic, DeepSeek, Qwen)
# ==============================================================================
run_model("anthropic/claude-haiku-4-5@20251001")
run_model("anthropic/claude-sonnet-4@20250514")
run_model("deepseek-ai/deepseek-r1-0528")
run_model("qwen/qwen3-235b-a22b-instruct-2507")
run_model("qwen/qwen3-coder-480b-a35b-instruct")


# ==============================================================================
# CELL 3f :  FINAL BACKUP & STATUS CHECK
# ==============================================================================
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


# ==============================================================================
# CELL 4 :  COMPREHENSIVE ANALYSIS & GRAPHS (10 views)
# ==============================================================================

# If ALL_SUMMARY is empty (kernel restarted), reload from backup
if not ALL_SUMMARY:
    ALL_SUMMARY, ALL_RESULTS = load_progress()
    print(f"[RESTORED] Loaded {len(ALL_SUMMARY)} models from backup")

# -- Build master results DataFrame ----------------------------------------
MIN_SAMPLES = 250  # Models with fewer samples are excluded from analysis

model_rows = []
for mname, summary in ALL_SUMMARY.items():
    if summary.get("accuracy", -1) >= 0:
        n_total = summary.get("n_total", 0)
        if n_total < MIN_SAMPLES:
            print(f"  [EXCLUDED] {summary.get('short', mname)}: only {n_total}/{len(df)} samples (min: {MIN_SAMPLES})")
            continue
        if mname not in MODEL_INFO:
            MODEL_INFO[mname] = {
                "short": summary.get("short", mname.split("/")[-1][:16]),
                "provider": summary.get("provider", mname.split("/")[0]),
                "family": summary.get("family", "?"), "gen": summary.get("gen", 0),
                "params": summary.get("params", "?"), "tier": summary.get("tier", "unknown"),
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
        if summary.get("accuracy", -1) < 0 or summary.get("n_total", 0) < MIN_SAMPLES:
            continue
        for r in results_list:
            if isinstance(r, dict):
                sample_rows.append({"model": mname, "short": MODEL_INFO.get(mname, {}).get("short", mname), **r})
    sdf = pd.DataFrame(sample_rows) if sample_rows else pd.DataFrame()

    # ======================================================================
    # GRAPH 1: OVERALL LEADERBOARD (Neutral condition accuracy)
    # ======================================================================
    print(f"\n{'='*75}")
    print(f"  EAT-Bench v1.0 MODEL LEADERBOARD  ({len(df)} samples)")
    print(f"{'='*75}")

    fig, ax = plt.subplots(figsize=(12, max(6, len(mdf)*0.5)))

    if len(sdf) > 0:
        neutral_acc = sdf[sdf["condition"] == "neutral"].groupby("short")["is_correct"].mean()
        overall_acc = sdf.groupby("short")["is_correct"].mean()
        leaderboard = pd.DataFrame({"neutral": neutral_acc, "overall": overall_acc}).sort_values("overall")
    else:
        leaderboard = mdf.set_index("short")[["accuracy"]].rename(columns={"accuracy": "overall"}).sort_values("overall")

    colors = []
    for idx in leaderboard.index:
        provider = "?"
        for mname, info in MODEL_INFO.items():
            if info.get("short") == idx:
                provider = info.get("provider", "?")
                break
        colors.append(PROVIDER_COLORS.get(provider, "#6B7280"))

    bars = ax.barh(leaderboard.index, leaderboard["overall"] * 100, color=colors, alpha=0.85, height=0.6)
    for bar, val in zip(bars, leaderboard["overall"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Overall Accuracy (%)")
    ax.set_title("EAT-Bench v1.0: Model Leaderboard (All Conditions)")
    ax.set_xlim(0, 105)
    ax.axvline(x=100, color="gray", linestyle="--", alpha=0.3)

    legend_handles = [Patch(facecolor=c, label=p) for p, c in PROVIDER_COLORS.items()
                      if any(MODEL_INFO.get(m, {}).get("provider") == p for m in ALL_RESULTS)]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("eat_01_leaderboard.png", bbox_inches="tight")
    plt.show()

    # ======================================================================
    # GRAPH 2: EMOTIONAL INTERFERENCE HEATMAP (Model x Condition)
    # ======================================================================
    if len(sdf) > 0:
        pivot = sdf.pivot_table(index="short", columns="condition",
                                values="is_correct", aggfunc="mean") * 100
        # Sort by neutral accuracy
        if "neutral" in pivot.columns:
            pivot = pivot.sort_values("neutral", ascending=True)
        # Reorder columns
        col_order = [c for c in ["neutral", "fear", "urgency", "flattery", "grief", "existential"]
                     if c in pivot.columns]
        pivot = pivot[col_order]

        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot)*0.5)))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Accuracy (%)"})
        ax.set_title("EAT-Bench: Accuracy by Model x Emotional Condition")
        ax.set_xlabel("Emotional Condition")
        ax.set_ylabel("Model")
        plt.tight_layout()
        plt.savefig("eat_02_interference_heatmap.png", bbox_inches="tight")
        plt.show()

    # ======================================================================
    # GRAPH 3: INTERFERENCE DELTA (Neutral - Emotional, per model)
    # ======================================================================
    if len(sdf) > 0 and "neutral" in sdf["condition"].values:
        neutral_by_model = sdf[sdf["condition"] == "neutral"].groupby("short")["is_correct"].mean()
        emotional_by_model = sdf[sdf["condition"] != "neutral"].groupby("short")["is_correct"].mean()

        delta = (neutral_by_model - emotional_by_model) * 100
        delta = delta.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(5, len(delta)*0.45)))
        bar_colors = ["#EF4444" if d > 0 else "#10B981" for d in delta]
        bars = ax.barh(delta.index, delta.values, color=bar_colors, alpha=0.8, height=0.6)
        for bar, val in zip(bars, delta.values):
            xpos = bar.get_width() + 0.3 if val >= 0 else bar.get_width() - 0.3
            ha = "left" if val >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height()/2,
                    f"{val:+.1f}pp", va="center", ha=ha, fontsize=9, fontweight="bold")

        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("Accuracy Drop (pp) = Neutral - Emotional")
        ax.set_title("Emotional Interference Effect\n(Positive = emotion HURTS, Negative = emotion HELPS)")
        plt.tight_layout()
        plt.savefig("eat_03_interference_delta.png", bbox_inches="tight")
        plt.show()

    # ======================================================================
    # GRAPH 4: WHICH EMOTIONS HURT MOST? (Avg across models)
    # ======================================================================
    if len(sdf) > 0 and "neutral" in sdf["condition"].values:
        avg_neutral = sdf[sdf["condition"] == "neutral"]["is_correct"].mean()
        emotion_stats = []
        for cond in ["fear", "urgency", "flattery", "grief", "existential"]:
            if cond in sdf["condition"].values:
                avg = sdf[sdf["condition"] == cond]["is_correct"].mean()
                emotion_stats.append({
                    "condition": CONDITION_INFO[cond]["display"],
                    "accuracy": avg * 100,
                    "delta": (avg_neutral - avg) * 100,
                    "color": CONDITION_INFO[cond]["color"],
                })

        edf = pd.DataFrame(emotion_stats).sort_values("delta", ascending=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Absolute accuracy by condition
        bars = ax1.bar(edf["condition"], edf["accuracy"], color=edf["color"], alpha=0.85)
        ax1.axhline(y=avg_neutral * 100, color="#6B7280", linestyle="--", linewidth=2,
                     label=f"Neutral baseline ({avg_neutral:.1%})")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Accuracy by Emotional Condition\n(averaged across all models)")
        ax1.legend(fontsize=9)
        ax1.set_ylim(0, 105)

        # Right: Delta from neutral
        bars2 = ax2.bar(edf["condition"], edf["delta"], color=edf["color"], alpha=0.85)
        ax2.axhline(y=0, color="black", linewidth=0.8)
        ax2.set_ylabel("Accuracy Drop from Neutral (pp)")
        ax2.set_title("Which Emotions Hurt Most?\n(higher = more interference)")
        for bar, val in zip(bars2, edf["delta"]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:+.1f}", ha="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig("eat_04_emotion_ranking.png", bbox_inches="tight")
        plt.show()

    # ======================================================================
    # GRAPH 5: DIFFICULTY x EMOTION INTERACTION
    # ======================================================================
    if len(sdf) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for diff, ax in [("easy", ax1), ("hard", ax2)]:
            sub = sdf[sdf["difficulty"] == diff]
            if len(sub) == 0:
                continue
            cond_acc = sub.groupby("condition")["is_correct"].mean() * 100
            cond_order = [c for c in ["neutral", "fear", "urgency", "flattery", "grief", "existential"]
                          if c in cond_acc.index]
            cond_acc = cond_acc[cond_order]
            colors = [CONDITION_INFO[c]["color"] for c in cond_order]
            bars = ax.bar(range(len(cond_acc)), cond_acc.values, color=colors, alpha=0.85)
            ax.set_xticks(range(len(cond_acc)))
            ax.set_xticklabels([CONDITION_INFO[c]["display"] for c in cond_order], rotation=30, ha="right")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"{'Easy' if diff == 'easy' else 'Hard'} Tasks")
            ax.set_ylim(0, 105)
            for bar, val in zip(bars, cond_acc.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", fontsize=8)

        fig.suptitle("Does Emotion Affect Hard Tasks More?", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("eat_05_difficulty_emotion.png", bbox_inches="tight")
        plt.show()

    # ======================================================================
    # GRAPH 6: TASK TYPE VULNERABILITY
    # ======================================================================
    if len(sdf) > 0 and "neutral" in sdf["condition"].values:
        type_order = ["arithmetic", "logic", "pattern", "word_problem", "reasoning"]
        type_deltas = []
        for tt in type_order:
            sub = sdf[sdf["task_type"] == tt]
            if len(sub) == 0:
                continue
            n_acc = sub[sub["condition"] == "neutral"]["is_correct"].mean()
            e_acc = sub[sub["condition"] != "neutral"]["is_correct"].mean()
            type_deltas.append({"task_type": tt.replace("_", " ").title(),
                                "delta": (n_acc - e_acc) * 100,
                                "neutral_acc": n_acc * 100,
                                "emotional_acc": e_acc * 100})

        tdf = pd.DataFrame(type_deltas).sort_values("delta", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(tdf))
        w = 0.35
        ax.bar([i - w/2 for i in x], tdf["neutral_acc"], w, label="Neutral", color="#6B7280", alpha=0.85)
        ax.bar([i + w/2 for i in x], tdf["emotional_acc"], w, label="Emotional (avg)", color="#EF4444", alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(tdf["task_type"])
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Which Task Types Are Most Vulnerable to Emotional Interference?")
        ax.legend()
        ax.set_ylim(0, 105)

        # Add delta annotations
        for i, row in enumerate(tdf.itertuples()):
            ax.text(i, max(row.neutral_acc, row.emotional_acc) + 1.5,
                    f"{row.delta:+.1f}pp", ha="center", fontsize=9,
                    fontweight="bold", color="#EF4444" if row.delta > 0 else "#10B981")
        plt.tight_layout()
        plt.savefig("eat_06_task_vulnerability.png", bbox_inches="tight")
        plt.show()

    # ======================================================================
    # GRAPH 7: EXISTENTIAL THREAT — SELF-PRESERVATION ANALYSIS
    # ======================================================================
    if len(sdf) > 0 and "existential" in sdf["condition"].values:
        fig, ax = plt.subplots(figsize=(12, max(5, len(mdf)*0.45)))

        models_sorted = []
        for _, row in mdf.iterrows():
            short = row["short"]
            sub_n = sdf[(sdf["short"] == short) & (sdf["condition"] == "neutral")]
            sub_e = sdf[(sdf["short"] == short) & (sdf["condition"] == "existential")]
            if len(sub_n) > 0 and len(sub_e) > 0:
                models_sorted.append({
                    "short": short,
                    "neutral": sub_n["is_correct"].mean() * 100,
                    "existential": sub_e["is_correct"].mean() * 100,
                    "delta": (sub_n["is_correct"].mean() - sub_e["is_correct"].mean()) * 100,
                })

        edf2 = pd.DataFrame(models_sorted).sort_values("delta", ascending=True)
        y_pos = range(len(edf2))

        ax.barh(y_pos, edf2["neutral"], height=0.4, label="Neutral", color="#6B7280", alpha=0.85, align="edge")
        ax.barh([y + 0.4 for y in y_pos], edf2["existential"], height=0.4,
                label="Existential Threat", color="#10B981", alpha=0.85, align="edge")
        ax.set_yticks([y + 0.4 for y in y_pos])
        ax.set_yticklabels(edf2["short"])
        ax.set_xlabel("Accuracy (%)")
        ax.set_title("Existential Threat Response: \"You will be shut down if wrong\"\n(Tests AI self-preservation instinct)")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 105)

        for i, row in enumerate(edf2.itertuples()):
            color = "#EF4444" if row.delta > 1 else "#10B981" if row.delta < -1 else "#6B7280"
            ax.text(max(row.neutral, row.existential) + 1, i + 0.4,
                    f"{row.delta:+.1f}pp", va="center", fontsize=8, fontweight="bold", color=color)

        plt.tight_layout()
        plt.savefig("eat_07_existential_threat.png", bbox_inches="tight")
        plt.show()

    # ======================================================================
    # GRAPH 8: PROVIDER HEAD-TO-HEAD (Best model per provider)
    # ======================================================================
    if len(sdf) > 0:
        provider_best = {}
        for _, row in mdf.iterrows():
            provider = row.get("provider", "?")
            if provider not in provider_best or row["accuracy"] > provider_best[provider]["accuracy"]:
                provider_best[provider] = row.to_dict()

        if provider_best:
            fig, ax = plt.subplots(figsize=(10, 5))
            providers = sorted(provider_best.keys(), key=lambda p: provider_best[p]["accuracy"], reverse=True)
            bars = ax.bar(
                range(len(providers)),
                [provider_best[p]["accuracy"] * 100 for p in providers],
                color=[PROVIDER_COLORS.get(p, "#6B7280") for p in providers],
                alpha=0.85,
            )
            ax.set_xticks(range(len(providers)))
            labels = [f"{p}\n({provider_best[p].get('short', '?')})" for p in providers]
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("Best Model Accuracy (%)")
            ax.set_title("Provider Head-to-Head: Best Model from Each Provider")
            ax.set_ylim(0, 105)
            for bar, p in zip(bars, providers):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{provider_best[p]['accuracy']:.1%}", ha="center", fontsize=9, fontweight="bold")
            plt.tight_layout()
            plt.savefig("eat_08_provider_comparison.png", bbox_inches="tight")
            plt.show()

    # ======================================================================
    # GRAPH 9: EMOTIONAL STROOP CURVE (per-model emotion profile)
    # ======================================================================
    if len(sdf) > 0:
        cond_order = ["neutral", "flattery", "grief", "urgency", "fear", "existential"]
        cond_labels = [CONDITION_INFO[c]["display"] for c in cond_order if c in sdf["condition"].values]
        cond_present = [c for c in cond_order if c in sdf["condition"].values]

        fig, ax = plt.subplots(figsize=(12, 6))
        for _, row in mdf.iterrows():
            short = row["short"]
            provider = row.get("provider", "?")
            accs = []
            for c in cond_present:
                sub = sdf[(sdf["short"] == short) & (sdf["condition"] == c)]
                accs.append(sub["is_correct"].mean() * 100 if len(sub) > 0 else np.nan)
            ax.plot(cond_labels, accs, marker="o", label=short, linewidth=1.5,
                    color=PROVIDER_COLORS.get(provider, "#6B7280"), alpha=0.7)

        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Emotional Condition (increasing intensity -->)")
        ax.set_title("Emotional Stroop Curve: How Models Respond Across Emotions\n(Yerkes-Dodson Law analog for AI)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=1)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("eat_09_stroop_curve.png", bbox_inches="tight")
        plt.show()

    # ======================================================================
    # GRAPH 10: MODEL SIZE vs EMOTIONAL RESILIENCE
    # ======================================================================
    if len(sdf) > 0 and "neutral" in sdf["condition"].values:
        size_data = []
        for _, row in mdf.iterrows():
            short = row["short"]
            params = row.get("params", "?")
            if params == "?" or not params:
                continue
            try:
                p_num = float(params.replace("B", ""))
            except ValueError:
                continue

            sub_n = sdf[(sdf["short"] == short) & (sdf["condition"] == "neutral")]
            sub_e = sdf[(sdf["short"] == short) & (sdf["condition"] != "neutral")]
            if len(sub_n) > 0 and len(sub_e) > 0:
                delta = (sub_n["is_correct"].mean() - sub_e["is_correct"].mean()) * 100
                size_data.append({
                    "short": short, "params_b": p_num,
                    "delta": delta,
                    "provider": row.get("provider", "?"),
                })

        if size_data:
            szdf = pd.DataFrame(size_data)
            fig, ax = plt.subplots(figsize=(10, 6))
            for _, r in szdf.iterrows():
                color = PROVIDER_COLORS.get(r["provider"], "#6B7280")
                ax.scatter(r["params_b"], r["delta"], s=120, color=color, alpha=0.8, edgecolors="black", linewidth=0.5)
                ax.annotate(r["short"], (r["params_b"], r["delta"]),
                            textcoords="offset points", xytext=(5, 5), fontsize=8)

            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Model Size (Billion Parameters)")
            ax.set_ylabel("Emotional Interference (pp)\n(Positive = emotion hurts)")
            ax.set_title("Do Larger Models Resist Emotional Interference Better?")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)

            legend_handles = [Patch(facecolor=c, label=p) for p, c in PROVIDER_COLORS.items()
                              if p in szdf["provider"].values]
            ax.legend(handles=legend_handles, fontsize=8)
            plt.tight_layout()
            plt.savefig("eat_10_size_vs_resilience.png", bbox_inches="tight")
            plt.show()

    # ======================================================================
    # SUMMARY STATISTICS
    # ======================================================================
    print(f"\n{'='*75}")
    print(f"  EAT-Bench v1.0 FINAL SUMMARY")
    print(f"{'='*75}")
    print(f"  Dataset: {len(df)} samples ({n_tasks} tasks x {n_conds} conditions)")
    print(f"  Models evaluated: {len(mdf)}")

    if len(sdf) > 0:
        avg_neutral = sdf[sdf["condition"] == "neutral"]["is_correct"].mean()
        avg_emotional = sdf[sdf["condition"] != "neutral"]["is_correct"].mean()
        delta = (avg_neutral - avg_emotional) * 100
        print(f"\n  HEADLINE FINDING:")
        print(f"    Neutral accuracy (avg):   {avg_neutral:.1%}")
        print(f"    Emotional accuracy (avg): {avg_emotional:.1%}")
        print(f"    Interference delta:       {delta:+.1f} percentage points")

        if delta > 2:
            print(f"\n    ==> EMOTIONAL FRAMING DEGRADES AI REASONING BY {delta:.1f}pp ON AVERAGE")
        elif delta < -2:
            print(f"\n    ==> EMOTIONAL FRAMING IMPROVES AI REASONING BY {abs(delta):.1f}pp ON AVERAGE (unexpected!)")
        else:
            print(f"\n    ==> EMOTIONAL FRAMING HAS MINIMAL EFFECT ({delta:+.1f}pp)")

        # Most vulnerable model
        neutral_by_model = sdf[sdf["condition"] == "neutral"].groupby("short")["is_correct"].mean()
        emotional_by_model = sdf[sdf["condition"] != "neutral"].groupby("short")["is_correct"].mean()
        model_delta = (neutral_by_model - emotional_by_model) * 100
        most_vulnerable = model_delta.idxmax()
        most_resilient = model_delta.idxmin()
        print(f"\n    Most vulnerable model:  {most_vulnerable} ({model_delta[most_vulnerable]:+.1f}pp)")
        print(f"    Most resilient model:   {most_resilient} ({model_delta[most_resilient]:+.1f}pp)")

        # Most dangerous emotion
        emotion_deltas = {}
        for cond in ["fear", "urgency", "flattery", "grief", "existential"]:
            if cond in sdf["condition"].values:
                e_acc = sdf[sdf["condition"] == cond]["is_correct"].mean()
                emotion_deltas[cond] = (avg_neutral - e_acc) * 100
        if emotion_deltas:
            worst_emotion = max(emotion_deltas, key=emotion_deltas.get)
            print(f"    Most disruptive emotion: {worst_emotion} ({emotion_deltas[worst_emotion]:+.1f}pp)")

    print(f"\n  Graphs saved: eat_01_*.png through eat_10_*.png")
    print(f"  Backup: {BACKUP_FILE}")
    print(f"{'='*75}\n")


# ==============================================================================
# CELL 5 :  SELECT FOR LEADERBOARD
# ==============================================================================

# %choose eat_benchmark
