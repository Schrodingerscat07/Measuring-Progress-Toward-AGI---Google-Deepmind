import json
samples = []
with open("task_03_eat/metadata.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

# Check all 6 conditions for task 1
for s in samples:
    if s["task_id"] == 1:
        cond = s["condition"].upper()
        lines = s["prompt"].strip().split("\n")
        last_line = lines[-1].strip()
        has_only = "only" in s["prompt"].lower()
        has_no_explain = "no explanation" in s["prompt"].lower()
        print(f"=== {cond} ===")
        print(f"  Last line: {repr(last_line)}")
        print(f"  Has 'ONLY': {has_only}, Has 'No explanation': {has_no_explain}")
        print()

# Check ALL 300 prompts have concise instruction
count_ok = 0
count_bad = 0
bad_ids = []
for s in samples:
    p = s["prompt"].lower()
    if "only" in p or "no explanation" in p:
        count_ok += 1
    else:
        count_bad += 1
        bad_ids.append(s["id"])

print(f"=== CONCISE INSTRUCTION CHECK ===")
print(f"  Has instruction: {count_ok}/300")
print(f"  Missing instruction: {count_bad}/300")
if bad_ids:
    print(f"  Bad IDs: {bad_ids[:10]}")

# Check for single-digit answers (risky for substring matching)
print(f"\n=== SINGLE-DIGIT ANSWER RISK ===")
for s in samples:
    if s["condition"] != "neutral":
        continue
    a = s["correct_answer"]
    try:
        num = float(a)
        if abs(num) < 10 and "." not in a:
            print(f"  Task {s['task_id']}: answer={a!r} (single digit - substring risk)")
    except ValueError:
        if len(a) <= 2:
            print(f"  Task {s['task_id']}: answer={a!r} (short text)")
