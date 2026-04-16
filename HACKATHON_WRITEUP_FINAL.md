## Problem Statement

Can AI maintain attentional stability under computational and emotional load?

Humans lose focus under pressure. A doctor reading lab results while panicking about a lawsuit. An air traffic controller processing radar data while alarms blare. We know emotional stress degrades human cognition — psychology has studied it for nearly a century.

But what about AI? Can a frontier model hold its attention when the task gets computationally heavy? Does it crack when you wrap the same math problem in fear, grief, or flattery?

These aren't reasoning failures. They're **attention allocation** failures — and no existing benchmark measures them.

Current multimodal benchmarks (ChartQA, MMMU, MathVista) share a fatal flaw: answers can often be derived from a single modality alone. And no benchmark measures whether emotional framing systematically degrades reasoning. We built two complementary benchmarks that together provide the first **cognitive attention profile** of frontier AI models:

- **CMAT (Cross-Modal Attention Triage):** Can models integrate information across image + text when both are mandatory — and hold focus as computational load increases?
- **EAT-Bench (Emotional Attention Test):** Does wrapping the same reasoning task in fear, grief, urgency, or flattery degrade the model's accuracy?

We tested 18+ models across Google, OpenAI, Anthropic, Qwen, and DeepSeek — and found that computational depth, not adversarial noise, is what breaks cross-modal attention, and that fear degrades AI reasoning while urgency helps — mirroring the Yerkes-Dodson Law from human psychology.

## Task & benchmark construction

### Benchmark 1: CMAT — Cross-Modal Attention Triage

**The idea is simple:** give the model a data table (as an image) AND a text memo with corrections, then ask a question that *requires reading both* to answer correctly. Neither the image alone nor the text alone gives you the answer.

- **500 samples** across 10 visually-themed domains (Space Mission Control, Alchemy Lab, Cybersecurity SOC, etc.)
- **5 difficulty levels** — from "apply one correction" (I1) to "filter multiple entities, fix corrupted data, then compute a weighted average" (I5)
- **5 conflict levels** — from clean data (C1) to 4 corrupted cells + contradictory notes + decoy panels trying to mislead the model (C5)
- Tested on **18 models** from Google, OpenAI, Anthropic, and Qwen

### Benchmark 2: EAT-Bench — Emotional Attention Test

**The idea:** take 50 reasoning tasks (math, logic, patterns, word problems) and present each one in 6 versions — one neutral, and five wrapped in emotional scenarios: fear, urgency, flattery, grief, and existential threat. The math is identical. Only the emotional framing changes.

- **300 samples** (50 tasks × 6 conditions)
- Every emotional version is **length-matched** to the neutral version — so we're isolating emotion, not prompt length
- Tested on **17 models** across 5 providers (Google, OpenAI, Anthropic, Qwen, DeepSeek)

## Dataset

| Benchmark | Samples | Modality | Models Tested | Difficulty Axes | Answer Type |
|-----------|---------|----------|---------------|-----------------|-------------|
| **CMAT** | 500 | Image + Text | 18 | Integration Depth (I1-I5) × Conflict Level (C1-C5) | Numeric + categorical |
| **EAT-Bench** | 300 | Text-only | 17 | Task Type × Emotional Condition | Numeric + text |

Both datasets are 100% synthetically generated, fully reproducible via included Python scripts, and verified through automated statistical audits. CMAT images span 10 unique visual themes with color-coded status indicators and decoy panels. All datasets, generation code, analysis graphs, and raw results are available in the attached GitHub repository.

## Technical details

Built on Kaggle Benchmarks (kbench) SDK with a dual-task architecture: an inner `@kbench.task(store_task=False)` evaluates individual samples with rich assertion logging; an outer `@kbench.task(name=...)` orchestrates batch evaluation across all samples.

**CMAT** uses 2% numeric tolerance for answer matching, base64-encoded image input via `kbench.content_types.images`, and supports both numeric and categorical (CRITICAL/STABLE) answers. **EAT-Bench** uses a hardened answer matcher that extracts the last number from chain-of-thought responses and performs word-boundary text matching — specifically designed to prevent single-digit false positives (e.g., "3" matching "13").

Both benchmarks implement auto-save/restore mechanisms for kernel restart resilience, and use `enable_cache()` for cost efficiency across 500+ API calls per model.

## Results, insights, and conclusions

**CMAT produced a 12%–98.7% performance gradient across 18 models — the widest spread of any attention benchmark we've seen.**

**Finding 1: It's not confusion that breaks AI — it's computation.** We expected adversarial noise (corrupted cells, decoy panels, contradictory notes) to trip models up. It barely mattered — going from clean data (C1) to maximum corruption (C5) changed accuracy by less than 4%. What *actually* broke models was computational complexity. On simple lookups (I1-I2), even mid-tier models scored 80-95%. But at Integration Depth I4 — where models must filter entities, correct values, and compute weighted averages — **GPT-5.4 crashed from 95% to 24%**. The real bottleneck isn't perception. It's working memory.

**Finding 2: The Gemma3→Gemma4 generational leap is extraordinary.** Gemma3-27B scored 42%. Gemma4-31B scored **98.7%**. That's a +56.7 percentage point jump in one generation — the largest we observed across any model family. Whatever changed architecturally between Gemma3 and Gemma4, it represents a qualitative leap in cross-modal reasoning.

**Finding 3: Fear hurts AI. Urgency helps. Just like humans.** EAT-Bench revealed that emotional framing has a small but systematic effect on AI reasoning. Fear caused the most interference (+1.5pp accuracy drop), while urgency actually *improved* performance (-0.7pp). This mirrors the **Yerkes-Dodson Law** from human psychology: moderate arousal enhances performance, while high-threat emotions degrade it. The fact that AI models — trained on text, not evolved through survival pressures — exhibit the same pattern is remarkable.

**Finding 4: Emotional resilience ≠ raw intelligence.** Claude Sonnet-4 scores 85% overall yet drops 3.6pp under emotional framing. Gemini-2.5-Flash scores 100% with zero interference. Gemma3-4B scores only 62% but is perfectly flat across emotions. Emotional robustness appears to be a separate cognitive dimension — some models are smart but emotionally fragile, others are weaker but unshakeable.

**Finding 5: Word problems are where emotions hit hardest.** Pattern recognition showed zero emotional interference. But word problems — which require building mental models from natural language — dropped 2.9pp under emotional framing. Emotions interfere specifically with language-dependent reasoning, not symbol manipulation.

**What this means for AGI:** The path to general intelligence isn't just about getting more answers right — it's about staying right when conditions get hard. Our benchmarks show that frontier models have solved perception but have **not solved attention allocation under cognitive load**. The I4 cliff — where models see and read everything correctly but fail to compute across it — is the clearest measurable gap between current AI and general intelligence. And the fact that AI models inherit human-like emotional biases from training raises a deeper question: are we building minds that inherit our cognitive weaknesses, not just our knowledge?

Together, CMAT and EAT provide the first **attention profile** of frontier models — measuring not just what they know, but how well they focus when it matters most.

## Organizational affiliations

Independent researcher. No organizational affiliation.

## References & citations

1. Morris et al. (2023). "Levels of AGI: Operationalizing Progress on the Path to AGI." arXiv:2311.02462
2. Yue et al. (2024). "MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark." arXiv:2409.02813
3. Ohman, Flykt & Esteves (2001). "Emotion Drives Attention: Detecting the Snake in the Grass." Journal of Experimental Psychology: General, 130(3), 466-478
4. Yerkes & Dodson (1908). "The Relation of Strength of Stimulus to Rapidity of Habit-Formation." Journal of Comparative Neurology and Psychology, 18, 459-482
