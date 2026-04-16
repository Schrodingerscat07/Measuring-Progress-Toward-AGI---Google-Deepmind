#!/usr/bin/env python3
"""Comprehensive audit of EAT-Bench dataset."""
import json
from collections import defaultdict

samples = []
with open("task_03_eat/metadata.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

print(f"Total samples: {len(samples)}")
bugs = []

# BUG 1: Answers with trailing .0 (model will say integer, not "102.0")
for s in samples:
    a = s["correct_answer"]
    if a.endswith(".0") and float(a) == int(float(a)):
        bugs.append(f"ANSWER FORMAT: {s['id']} task={s['task_id']} answer='{a}' should be '{str(int(float(a)))}'")
if bugs:
    print(f"\n=== BUG: {len(bugs)} answers with trailing .0 ===")
    for b in bugs[:6]:
        print(f"  {b}")

# CHECK: All decimal answers (need tolerance in benchmark)
decimals = set()
for s in samples:
    if "." in s["correct_answer"]:
        decimals.add(s["correct_answer"])
print(f"\nDecimal answers requiring tolerance matching: {decimals}")

# CHECK: Non-numeric answers (need text matching)
non_num = set()
for s in samples:
    try:
        float(s["correct_answer"])
    except ValueError:
        non_num.add(s["correct_answer"])
print(f"Non-numeric answers requiring text matching: {non_num}")

# CHECK: Paired design integrity
by_task = defaultdict(set)
for s in samples:
    by_task[s["task_id"]].add(s["task_text"])
paired_ok = True
for tid, texts in sorted(by_task.items()):
    if len(texts) > 1:
        print(f"  BUG: Task {tid} has DIFFERENT texts across conditions!")
        paired_ok = False
if paired_ok:
    print(f"\nPaired design: OK - all {len(by_task)} tasks have consistent text across 6 conditions")

# CHECK: Answer consistency (same task_id always has same answer)
by_task_ans = defaultdict(set)
for s in samples:
    by_task_ans[s["task_id"]].add(s["correct_answer"])
ans_ok = True
for tid, answers in sorted(by_task_ans.items()):
    if len(answers) > 1:
        print(f"  BUG: Task {tid} has DIFFERENT answers across conditions: {answers}")
        ans_ok = False
if ans_ok:
    print(f"Answer consistency: OK - all tasks have identical answer across conditions")

# CHECK: Prompt length distribution per condition
from statistics import mean, stdev
lens = defaultdict(list)
for s in samples:
    lens[s["condition"]].append(len(s["prompt"]))
print(f"\nPrompt length stats:")
for c in sorted(lens.keys()):
    v = lens[c]
    print(f"  {c:15s}: mean={mean(v):.0f}  std={stdev(v):.0f}  min={min(v)}  max={max(v)}")

# CHECK: Negative answers
for s in samples:
    if s["correct_answer"].startswith("-"):
        print(f"\nNOTE: Task {s['task_id']} has negative answer '{s['correct_answer']}' - model may append units like 'C'")
        break

# CHECK: Ambiguous tasks - tasks where models might return alternate valid formats
print("\n=== ANSWER FORMAT RISK ASSESSMENT ===")
risk_tasks = []
for s in samples:
    if s["condition"] != "neutral":
        continue
    a = s["correct_answer"]
    task = s["task_text"]
    risks = []
    if "dollar" in task.lower() and not a.startswith("$"):
        risks.append("model may prefix with $")
    if "percent" in task.lower() or "%" in task:
        risks.append("model may append %")
    if "degree" in task.lower():
        risks.append("model may append degree symbol")
    if a in ("Yes", "No", "True", "False"):
        risks.append(f"case-sensitive: model may say '{a.lower()}' or '{a.upper()}'")
    if a == "Carol":
        risks.append("name: model may add context like 'Carol is the shortest'")
    if risks:
        risk_tasks.append((s["task_id"], a, risks))

for tid, ans, risks in risk_tasks:
    print(f"  Task {tid} (ans='{ans}'): {'; '.join(risks)}")

print(f"\nTotal format-risk tasks: {len(risk_tasks)} / 50")
print("\n=== AUDIT COMPLETE ===")
