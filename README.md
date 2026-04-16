# Attention Under Pressure: Profiling What Breaks — and What Bends — AI Focus

**Kaggle Hackathon: Measuring Progress Toward AGI – Cognitive Abilities**  
**Track: Attention** | **Team Batman** — Shiva Sai ([@shivasai77](https://www.kaggle.com/shivasai77))

🔗 **[Kaggle Benchmark](https://www.kaggle.com/benchmarks/shivasai77/attention-under-pressure)** | 📄 **[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups)**

---

## Overview

Two novel benchmarks that together provide the first **cognitive attention profile** of frontier AI models:

| Benchmark | What It Tests | Samples | Models | Key Finding |
|-----------|--------------|---------|--------|-------------|
| **CMAT** | Cross-modal integration (image + text) | 500 | 18 | Computational depth breaks attention (I4 cliff: 95% → 24%) |
| **EAT-Bench** | Emotional interference on reasoning | 300 | 17 | Fear hurts (+1.5pp), urgency helps (-0.7pp) — mirrors human Yerkes-Dodson Law |

## Results at a Glance

### CMAT: 12% → 98.7% Performance Gradient
![CMAT Leaderboard](results/cmat_graphs/graph_leaderboard.png)

### EAT: Emotional Interference is Small but Systematic
![EAT Interference](results/eat_graphs/eat_03_interference_delta.png)

## Repository Structure

```
├── task_02_cmat/                    # CMAT dataset (500 samples)
│   ├── metadata.jsonl               # Sample metadata
│   ├── images/                      # 500 dashboard images (10 themed domains)
│   ├── generate_cmat_v2.py          # Reproducible dataset generator
│   └── dataset_card.md              # Dataset documentation
│
├── task_03_eat/                     # EAT-Bench dataset (300 samples)
│   ├── metadata.jsonl               # Sample metadata
│   ├── generate_eat.py              # Reproducible dataset generator
│   └── dataset_card.md              # Dataset documentation
│
├── benchmark_cmat_clean.py          # CMAT benchmark code (Kaggle notebook)
├── benchmark_eat_clean.py           # EAT benchmark code (Kaggle notebook)
├── benchmark_cmat_task.py           # Full CMAT code with graphs + analysis
├── benchmark_eat_task.py            # Full EAT code with graphs + analysis
│
├── results/                         # All outputs from benchmark runs
│   ├── cmat_graphs/                 # 10 CMAT analysis graphs
│   ├── eat_graphs/                  # 10 EAT analysis graphs  
│   ├── cmat_results_backup.json     # Raw CMAT results (18 models × 500 samples)
│   └── eat_results_backup.json      # Raw EAT results (17 models × 300 samples)
│
├── HACKATHON_WRITEUP_FINAL.md       # Submission writeup (<1,500 words)
└── agents.md                        # Project history & agent notes
```

## Key Findings

### CMAT — The Computational Cliff
- Models scoring 90%+ on simple lookups **crash to 15-30%** on multi-step aggregation (I4)
- Adversarial noise (C1→C5) barely affects accuracy (<4% change)
- **Gemma3→Gemma4 leap**: 42% → 98.7% (+56.7pp) — largest generational jump observed

### EAT-Bench — The Emotional Profile
- **Fear** is the most disruptive emotion (+1.5pp interference)
- **Urgency** slightly *helps* performance (-0.7pp) — mirrors human arousal research
- **Sonnet-4** and **Qwen3-235B** are the most emotionally vulnerable despite high overall accuracy
- **Word problems** are the most susceptible task type (+2.9pp)
- Top models (Gem2.5-F, Gem3.1-P) show **zero** emotional interference

## Reproducing Results

### CMAT
```bash
# Generate the dataset
cd task_02_cmat && python generate_cmat_v2.py

# Run on Kaggle: paste benchmark_cmat_clean.py into a notebook
# Add cmat-v1-0 dataset → Save & Run All
```

### EAT-Bench
```bash
# Generate the dataset  
cd task_03_eat && python generate_eat.py

# Run on Kaggle: paste benchmark_eat_clean.py into a notebook
# Add eat-bench-v1 dataset → Save & Run All
```

## License
Apache 2.0
