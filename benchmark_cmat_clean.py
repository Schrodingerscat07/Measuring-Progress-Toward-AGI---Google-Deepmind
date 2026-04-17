"""
CMAT v2.0 — Cross-Modal Attention Triage — BENCHMARK VERSION
=============================================================
Minimal version for running from the Kaggle Benchmark page.
Only contains: data loading + task definitions + %choose
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1: SETUP & DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

import os, json, base64, re
import pandas as pd
import kaggle_benchmarks as kbench
from kaggle_benchmarks.content_types import images
from kaggle_benchmarks.assertions import AssertionResult

os.environ["RENDER_SUBRUNS"] = "False"

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

print(f"[OK] Loaded {len(df)} samples | Domains: {df['domain'].nunique()}")

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


# ══════════════════════════════════════════════════════════════════════════════
# CELL 2: KBENCH TASK DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

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

    final_assertion = check_cross_modal(sample_id=id, correct_answer=correct_answer,
                      image_only_trap=image_only_trap, domain=domain,
                      integration_depth=integration_depth,
                      conflict_level=conflict_level,
                      difficulty_cell=difficulty_cell,
                      model_response=model_response)

    is_correct = answers_match(model_response, correct_answer)
    hit_trap = (not is_correct) and answers_match(model_response, image_only_trap)

    SAMPLE_COLLECTOR.append({
        "id": id, "correct_answer": correct_answer,
        "predicted": model_response.strip()[:200],
        "is_correct": is_correct, "hit_image_trap": hit_trap,
        "domain": domain, "integration_depth": integration_depth,
        "conflict_level": conflict_level, "difficulty_cell": difficulty_cell,
    })

    return final_assertion


@kbench.task(
    name="CMAT-v2.0",
    description=(
        "Cross-Modal Attention Triage v2.0: 500 samples across 10 domains. "
        "Tests whether models can integrate text memos + images to perform "
        "corrections, arithmetic, conditionals, aggregation, and multi-hop reasoning."
    ),
)
def cmat_benchmark(llm):
    SAMPLE_COLLECTOR.clear()

    with kbench.client.enable_cache():
        single_cmat_task.evaluate(
            stop_condition=lambda r: len(r) == df.shape[0],
            max_attempts=1, llm=[llm], evaluation_data=df,
            n_jobs=4, timeout=180, remove_run_files=False,
        )

    results = pd.DataFrame(SAMPLE_COLLECTOR)
    total = len(results)
    correct = int(results["is_correct"].sum()) if total > 0 else 0
    acc = correct / total if total > 0 else 0.0
    std = float(results["is_correct"].std()) if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  CMAT v2.0 RESULTS  |  Acc: {acc:.1%}  |  {correct}/{total}")
    print(f"{'='*60}\n")

    return kbench.assertions.assert_true(acc >= 0,
        expectation=f"Accuracy: {acc:.4f} +/- {std:.4f} on {total} samples")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3: RUN ONE MODEL TO GENERATE TASK + SELECT FOR BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

# Run one model to bootstrap the task (required before %choose works)
cmat_benchmark.run(kbench.llms["google/gemini-2.0-flash-lite"])

# %choose cmat_benchmark
