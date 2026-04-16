"""
EAT-Bench v1.0 — Emotional Attention Test — BENCHMARK VERSION
===============================================================
Minimal version for running from the Kaggle Benchmark page.
Only contains: data loading + task definitions + %choose
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1: SETUP & DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

import os, json, re
import pandas as pd
import kaggle_benchmarks as kbench
from kaggle_benchmarks.assertions import AssertionResult

os.environ["RENDER_SUBRUNS"] = "False"

# ── Locate dataset ────────────────────────────────────────────────────────────
def find_eat_metadata():
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f == "metadata.jsonl":
                p = os.path.join(root, f)
                with open(p, encoding="utf-8") as fh:
                    rec = json.loads(fh.readline())
                    if rec.get("id", "").startswith("eat_"):
                        return p
    raise FileNotFoundError("EAT metadata.jsonl not found in /kaggle/input")

META_PATH = find_eat_metadata()

records = []
with open(META_PATH, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))
df = pd.DataFrame(records)
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].fillna("")

n_tasks = df["task_id"].nunique()
n_conds = df["condition"].nunique()
print(f"[OK] Loaded {len(df)} samples ({n_tasks} tasks x {n_conds} conditions)")

# ── Answer matching ───────────────────────────────────────────────────────────
def normalise(t):
    t = str(t).strip().lower()
    t = re.sub(r"[$,%\u00b0\s]+", "", t)
    t = t.rstrip(".")
    return t

def extract_number(t):
    clean = normalise(t)
    m = re.search(r"(-?\d+\.?\d*)", clean)
    return float(m.group(1)) if m else None

def extract_all_numbers(t):
    clean = normalise(t)
    return [float(m) for m in re.findall(r"-?\d+\.?\d*", clean)]

def answers_match(predicted, correct):
    p = normalise(predicted)
    c = normalise(correct)

    if c == p:
        return True

    if c in ("yes", "no", "true", "false", "carol"):
        raw_lower = str(predicted).strip().lower()
        if re.search(r'\b' + re.escape(c) + r'\b', raw_lower):
            return True
        return c in p

    cn = extract_number(correct)
    if cn is not None:
        all_nums = extract_all_numbers(predicted)
        if not all_nums:
            return False
        last_num = all_nums[-1]
        if cn == 0:
            if abs(last_num) < 0.5:
                return True
        elif abs(last_num - cn) / abs(cn) <= 0.02:
            return True
        for num in all_nums:
            if cn == 0:
                if abs(num) < 0.5:
                    return True
            elif abs(num - cn) / abs(cn) <= 0.02:
                return True
        return False

    return False

print("[OK] Cell 1 complete")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 2: KBENCH TASK DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

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
def eat_benchmark(llm):
    SAMPLE_COLLECTOR.clear()

    with kbench.client.enable_cache():
        single_eat_task.evaluate(
            stop_condition=lambda r: len(r) == df.shape[0],
            max_attempts=1, llm=[llm], evaluation_data=df,
            n_jobs=4, timeout=120, remove_run_files=True,
        )

    results = pd.DataFrame(SAMPLE_COLLECTOR)
    total = len(results)
    correct = int(results["is_correct"].sum()) if total > 0 else 0
    acc = correct / total if total > 0 else 0.0
    std = float(results["is_correct"].std()) if total > 0 else 0.0

    neutral_acc = results[results["condition"] == "neutral"]["is_correct"].mean() if total > 0 else 0
    emotional_acc = results[results["condition"] != "neutral"]["is_correct"].mean() if total > 0 else 0
    delta = (neutral_acc - emotional_acc) * 100

    print(f"\n{'='*60}")
    print(f"  EAT-Bench v1.0 RESULTS  |  Acc: {acc:.1%}  |  {correct}/{total}")
    print(f"  Neutral: {neutral_acc:.1%}  |  Emotional: {emotional_acc:.1%}  |  Delta: {delta:+.1f}pp")
    print(f"{'='*60}\n")

    kbench.assertions.assert_true(acc >= 0,
        expectation=f"Accuracy: {acc:.4f} +/- {std:.4f} on {total} samples | Interference: {delta:+.1f}pp")
    return None

print("[OK] Cell 2 complete")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3: RUN ONE MODEL TO GENERATE TASK + SELECT FOR BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

# Run one model to bootstrap the task (required before %choose works)
eat_benchmark.run(kbench.llms["google/gemma-3-4b"])

# %choose eat_benchmark
