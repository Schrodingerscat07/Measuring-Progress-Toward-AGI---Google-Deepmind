# Attention Under Pressure: Profiling the Cognitive Resilience of Frontier AI

**A Comprehensive Benchmark for Measuring Progress Toward AGI (Attention & Emotional Reasoning)**

---

## 🏆 Project Overview

**Attention Under Pressure** is a research-driven benchmark suite built for the **Kaggle Google DeepMind Hackathon**. It profiles how well frontier AI models allocate their attention under two distinct types of stress: **Computational Load** and **Emotional Framing**.

We introduce two novel benchmarks that reveal the current gap between contemporary LLMs and true General Intelligence:

1.  **CMAT (Cross-Modal Attention Triage):** Tests if models can integrate complex information across a mandatory image-text pair while performing multi-hop reasoning.
2.  **EAT-Bench (Emotional Attention Test):** Measures the systematic interference caused by emotional wrappers (Fear, Urgency, Flattery) on identical logic tasks.

🔗 **[Kaggle Benchmark Suite](https://www.kaggle.com/benchmarks/shivasai77/attention-under-pressure)** | 📄 **[Final Research Writeup](HACKATHON_WRITEUP_FINAL.md)**

---

## 📊 Results At A Glance

### 1. The CMAT Gradient (12% → 98.7%)
Models exhibit a sharp "performance cliff" as computational depth increases. The bottleneck isn't reading the image—it's **computing** across modalities.

![CMAT Leaderboard](results/cmat_graphs/graph_leaderboard.png)

### 2. The EAT Interference Profile
Frontier models are surprisingly resilient, but specific emotions like **Fear** cause systematic reasoning degradation, while **Urgency** can actually improve performance.

![EAT Interference](results/eat_graphs/eat_03_interference_delta.png)

---

## 🧠 Key Research Findings

### 📉 Finding 1: The "I4 Cliff" — Computation Breaks Attention
In CMAT, models easily handle lookup tasks (I1-I2). However, when required to **aggregate** and **filter** data across modalities (Level I4), accuracy drops from 95% to **24%** for even top-tier models like GPT-5.4. This proves that "Attention" in current AI is limited by a working-memory bottleneck rather than a perceptual one.
![Integration Depth](results/cmat_graphs/graph_integration_depth.png)

### 📈 Finding 2: The Gemma Generational Leap
We observed a staggering **+56.7pp jump** between Gemma 3 (42%) and Gemma 4 (98.7%). This suggests a massive architectural improvement in how newer weights handle cross-modal binding.
![Gemma Progress](results/cmat_graphs/graph_provider_comparison.png)

### 😨 Finding 3: Fear vs. Urgency (The AI Yerkes-Dodson Law)
Mirroring human psychology, AI models show a "Stress Response" curve. **Fear** degrades reasoning across 75% of tested models, while **Urgency** wrappers consistently produced slightly more accurate results—suggesting models "focus" more under perceived time pressure.
![Emotion Ranking](results/eat_graphs/eat_04_emotion_ranking.png)

### 🧩 Finding 4: Emotional Resilience is a Unique Dimension
Performance on logic tasks ≠ Emotional Resilience. Some smart models (Sonnet-4) are highly susceptible to emotional framing (+3.6pp interference), while some smaller models (Gemini-Flash) are perfectly unshakeable.
![Size vs Resilience](results/eat_graphs/eat_10_size_vs_resilience.png)

---

## 📂 Repository Structure

```
├── task_02_cmat/          # CMAT Dataset (500 samples, 10 themed domains)
│   ├── images/            # Synthetic dashboard images
│   └── generate_cmat.py   # Dataset generator
│
├── task_03_eat/           # EAT-Bench Dataset (300 samples, 6 conditions)
│   └── generate_eat.py    # Dataset generator
│
├── results/               # Compiled analytics and metrics
│   ├── cmat_graphs/       # 10 Detailed cross-modal plots
│   └── eat_graphs/        # 10 Detailed emotional plots
│
├── benchmark_cmat_task.py # Full source for CMAT Evaluation
└── benchmark_eat_task.py  # Full source for EAT Evaluation
```

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/Schrodingerscat07/Attention-Under-Pressure.git
pip install kaggle-benchmarks
```

### 2. Re-generating Data
To generate the 500-sample CMAT dataset:
```bash
python task_02_cmat/generate_cmat_v2.py
```

### 3. Running the Benchmark
Paste the contents of `benchmark_cmat_clean.py` into a Kaggle Notebook and trigger the `.evaluate()` method.

---

## 🛡️ License & Acknowledgements
Built by **Team Batman** for the Kaggle "Measuring Progress Toward AGI" hackathon.  
Special thanks to the **Google DeepMind** team for the `kbench` SDK and the challenge of defining cognitive limits.

Licensed under **Apache 2.0**.
