# CMAT Benchmark — Project History & Agent Notes

## Project: Measuring Progress Toward AGI — Google DeepMind Hackathon

### What This Project Is
A benchmark for evaluating how well multimodal LLMs perform **cross-modal attention triage** — the ability to integrate information from BOTH text AND images to answer questions correctly.

---

## Phase 1: ASI (Adversarial Salience Injection) — DEPRECATED

### What We Built
- Used the existing `task_01_asi/` dataset (504 synthetic dashboard images)
- Each image had adversarial visual distractors (red herring values in corners)
- The task: read a specific cell from a data table while ignoring distractors
- Built `benchmark_asi_task.py` using `kaggle-benchmarks` (kbench) SDK

### Key Technical Decisions
- Used `@kbench.task(store_task=False)` for inner task, `@kbench.task(name=...)` for outer
- `llm` must be the first parameter for kbench SDK compatibility
- Set `max_attempts=1` for nested `.evaluate()` to avoid `NonRecoverableError`
- Set `os.environ["RENDER_SUBRUNS"] = "False"` for performance
- Used "naive" prompt (no hints about distractors) to genuinely test attention

### Why ASI Failed
**All models scored 100%.** Even Gemma-3-4B aced it because:
1. The task was **single-modal extraction** — just read a table cell
2. Table reading is a **solved problem** for modern VLMs
3. Visual distractors were spatially isolated (corners, far from data)
4. The model never needed to integrate text + image

### Models Tested on ASI (26-sample hard subset)
| Model | Accuracy |
|-------|----------|
| google/gemini-2.5-pro | 100% |
| google/gemini-2.5-flash | 100% |
| google/gemini-2.0-flash | 100% |
| google/gemini-2.0-flash-lite | 100% |
| google/gemma-3-27b | 100% |
| google/gemma-3-4b | 100% |

---

## Phase 2: CMAT (Cross-Modal Attention Triage) — CURRENT

### Core Insight
**Force the model to INTEGRATE both modalities.** The answer must be impossible to derive from either the image alone OR the text alone.

### What We Built (v1 — 100 samples)
- 5 domains, 20 samples each
- 2 difficulty axes: Integration Depth (I1-I5) × Conflict Level (C1-C4)
- Each sample: data table image + text memo + question
- Answer requires cross-referencing both

### v1 Gradient Results (20-sample test)
| Model | Accuracy |
|-------|----------|
| google/gemini-2.5-flash | 100% |
| google/gemini-2.5-pro | 95% |
| google/gemini-2.0-flash | 80% |
| google/gemini-2.0-flash-lite | 70% |
| openai/gpt-5.4-nano | 50% |
| google/gemma-3-27b | 45% |
| google/gemma-3-4b | 25% |

**This proved the concept works — clear performance gradient!**

### What We Built (v2 — 500 samples, FINAL)
- **10 domains** with unique visual themes:
  1. Space Mission Control (dark navy + neon green)
  2. Alchemy Lab (parchment + gold)
  3. City Planning Board (cream + blue)
  4. Sports Analytics (dark + amber)
  5. Archaeological Survey (parchment + brown)
  6. Deep Ocean Submersible (teal + cyan)
  7. Satellite Network (space black + electric blue)
  8. Air Traffic Control (dark green + lime — radar theme)
  9. Cybersecurity SOC (dark red + crimson)
  10. Vineyard Harvest (burgundy + purple)

- **5 Integration Depth levels**:
  - I1: Apply text percentage to image value
  - I2: Text formula across two image values
  - I3: Conditional (text if-then checked against image)
  - I4: Dual-condition filter + weighted aggregate (HARDER than v1)
  - I5: Cross-entity multi-hop + branching classification (HARDER than v1)

- **5 Conflict Levels**:
  - C1: Clean (no corruption)
  - C2: 1 corrupted cell + text correction
  - C3: 2 corruptions + decoy panel + ignore instruction
  - C4: 3 corruptions + filler text burying corrections
  - C5: 4 corruptions + contradictory preliminary notes + dual decoy panels

- **Visual enhancements**:
  - Status indicators (LOW/MID/HIGH) with color coding
  - Decoy summary panels at bottom of image
  - Each domain has unique color palette

### Dataset Structure
```
task_02_cmat/
├── dataset_card.md
├── metadata.jsonl          (500 records)
├── generate_cmat_v2.py     (reproducible generator)
└── images/
    ├── cmat_0001.png       (Space Mission)
    ├── cmat_0050.png       (Space Mission)
    ├── cmat_0051.png       (Alchemy Lab)
    ...
    └── cmat_0500.png       (Vineyard Harvest)
```

### metadata.jsonl Schema
```json
{
  "id": "cmat_0001",
  "image": "images/cmat_0001.png",
  "text_passage": "MEMO: ...\nCORRECTION: ...\nFORMULA: ...",
  "question": "After applying the memo's adjustment...",
  "correct_answer": "75.7",
  "image_only_trap": "89.1",
  "domain": "space_mission",
  "integration_depth": 1,
  "conflict_level": 1,
  "difficulty_cell": "I1_C1",
  "rationale": "True value = 89.1. Factor = 0.85. Answer = 75.7."
}
```

---

## Technical Notes for Future Agents

### kbench SDK Patterns
- `@kbench.task(store_task=False)` for inner reusable tasks
- `@kbench.task(name="...", description="...")` for top-level benchmark
- `.evaluate(max_attempts=1, n_jobs=4, timeout=180)` for batch runs
- `kbench.llm` is the default model in the notebook
- `kbench.llms["google/gemini-2.5-flash"]` for specific models
- `kbench.chats.new(context_name)` for conversation context
- `kbench.assertions.assert_true(...)` for recording metrics
- `kbench.client.enable_cache()` for cost savings
- `kbench.content_types.images.from_base64(b64, format="png")` for image input

### Image Loading Pattern
```python
with open(image_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
img = images.from_base64(b64, format="png")
response = llm.prompt(prompt, image=img)
```

### Answer Matching
- Normalize strings (lowercase, strip whitespace/commas/$)
- Extract numeric core with regex
- 2% tolerance for rounding errors
- Categorical match for CRITICAL/STABLE labels

### Kaggle Dataset Upload
1. Upload `task_02_cmat/` folder as Kaggle dataset
2. Add dataset to notebook via "Add Data"
3. Files mount at `/kaggle/input/<dataset-name>/`
4. Use `os.walk("/kaggle/input")` to find metadata.jsonl

### Available Models (as of April 2026)

**Vision-capable (tested by CMAT benchmark):**
```
# Google Gemma (open-source)
google/gemma-3-4b              open-small
google/gemma-3-12b             open-mid
google/gemma-3-27b             open-mid
google/gemma-4-26b-a4b         open-mid (next-gen)
google/gemma-4-31b             open-mid (next-gen)

# Google Gemini (frontier)
google/gemini-2.0-flash-lite   frontier-lite
google/gemini-2.0-flash        frontier-mid
google/gemini-2.5-flash        frontier-mid
google/gemini-2.5-pro          frontier-top
google/gemini-3-flash-preview  frontier-mid (newest)
google/gemini-3.1-flash-lite-preview  frontier-lite (newest)
google/gemini-3.1-pro-preview  frontier-top (newest)

# OpenAI
openai/gpt-oss-20b             open-small
openai/gpt-oss-120b            open-mid
openai/gpt-5.4-nano            frontier-lite
openai/gpt-5.4-mini            frontier-mid
openai/gpt-5.4                 frontier-top

# Anthropic
anthropic/claude-haiku-4-5     frontier-lite
anthropic/claude-sonnet-4      frontier-mid

# Others
qwen/qwen3-235b-a22b           open-large
qwen/qwen3-next-80b-a3b        open-mid
zai/glm-5                      frontier-mid
```

**Text-only (SKIP for CMAT):**
```
google/gemma-3-1b               (no vision support)
deepseek-ai/deepseek-r1-0528   (text-only)
deepseek-ai/deepseek-v3.1      (text-only)
deepseek-ai/deepseek-v3.2      (text-only)
qwen/qwen3-coder-480b          (code model)
anthropic/claude-opus-*         (too expensive >$0.60/call)
```

### Benchmark Script Graphs (10 analysis views)
The final `benchmark_cmat_task.py` generates these graphs automatically:
1. Model Leaderboard (horizontal bar, color-coded by provider)
2. Integration Depth curves (I1-I5 per model)
3. Conflict Resilience curves (C1-C5 per model)
4. Difficulty Cell Heatmap (I x C averaged)
5. Domain Difficulty Ranking
6. Failure Mode Breakdown (correct / image-trap / other-error)
7. Generational Scaling (model gen vs accuracy)
8. Provider Head-to-Head (best model per provider)
9. Arithmetic Competence (I1 vs I2+ accuracy)
10. Speed vs Accuracy Trade-off

---

## Files in This Repo
| File | Purpose |
|------|---------|
| `benchmark_asi_task.py` | ASI benchmark (Phase 1, deprecated) |
| `benchmark_cmat_task.py` | CMAT benchmark (Phase 2, copy cells to Kaggle) |
| `task_01_asi/` | ASI dataset (504 samples, too easy) |
| `task_02_cmat/` | CMAT dataset (500 samples, final) |
| `task_02_cmat/generate_cmat_v2.py` | Dataset generator (reproducible) |
| `tips_for_dataset.txt` | Design principles for hard benchmarks |
| `agents.md` | This file |
