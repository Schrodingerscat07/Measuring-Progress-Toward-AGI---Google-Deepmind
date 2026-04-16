# EAT-Bench: Emotional Attention Test
## Detailed Research Findings & Technical Analysis

The EAT-Bench framework investigates a dimension of AI reasoning previously ignored by traditional benchmarks: Emotional Robustness. By evaluating 17 models on 300 instances of paired logical tasks disguised in emotional wrappers, we observed human-like cognitive biases embedded within frontier LLMs.

### 1. The Yerkes-Dodson Law in Artificial Intelligence
The most profound finding of the EAT-Bench research is the identification of a systematic, predictable variance in AI reasoning accuracy depending on the assigned emotional valence. This mirrors the  psychological "Yerkes-Dodson Law", which states that moderate arousal improves cognitive performance, while high arousal (stress/fear) degrades it.

*   **Fear (High Threat, High Arousal):** Across the aggregated model panel, "Fear" scenarios (e.g., ticking bombs, hostage situations) caused the most significant degradation in reasoning accuracy, resulting in an average drop of **1.5 percentage points** compared to the neutral baseline.
*   **Urgency (Moderate Threat, Action-Oriented):** Scenarios framing the problem as time-sensitive but non-fatal (e.g., a stock market trade deadline) consistently *improved* average model performance by **0.7 percentage points**. 
*   **Conclusion:** LLMs absorb the semantic associations of human emotions present in their training corpora. When processing "fear" tokens, the model's latent representation shifts towards pathways associated with panic or irrationality in human text, subtly disrupting sequential logic circuits.

### 2. The Decoupling of "Intelligence" and "Emotional Resilience"
A prevailing assumption in AI scaling laws is that as models become more capable (higher general reasoning), they become more robust to prompt perturbations. EAT-Bench proves this is false for emotional framing.

*   **Claude 3 Sonnet-4:** This model represents top-tier frontier capability, scoring a baseline of 88% on the neutral tasks. However, under emotional conditions, its accuracy collapsed to 82% in some categories, yielding an average emotional interference penalty of **-3.6%**. It was the most emotionally vulnerable model tested.
*   **Gemini 2.5 Flash:** A mid-sized, highly optimized model that scored a perfect 100% on neutral tasks *and* maintained 100% across all emotional variants.
*   **Gemma 3-4B:** A small, lower-capability model scoring 62% baseline, but exhibiting zero emotional interference.
*   **Conclusion:** Emotional resilience is a distinct structural characteristic of a model's alignment and training data diet, not a byproduct of parameter count or raw reasoning capability. Highly capable models can still be highly susceptible to psychological framing.

### 3. Task-Type Vulnerability Profile
Emotional interference does not apply equally to all types of logic. We categorized the 50 deterministic tasks into specific mathematical and logical verticals.

*   **Word Problems (Highest Vulnerability):** Math embedded inside narrative scenarios suffered the highest degradation (**-2.9%**). Emotional text forces the model to heavily process the semantic context to extract variables, allowing the emotional tone to bleed into the mathematical extraction process.
*   **Logical Deductions (Moderate Vulnerability):** Tasks like knight/knave puzzles or sequencing showed moderate interference. 
*   **Pattern Recognition & Raw Arithmetic (Zero Vulnerability):** Tasks lacking a narrative structure (pure symbolic manipulation) demonstrated complete immunity to emotional preamble texts.
*   **Conclusion:** The vulnerability occurs during the language-to-symbol translation phase. Once variables are extracted into pure logic, the emotional disruption ceases.

### 4. Implications for AGI and Safety
As AI systems are increasingly deployed in high-stakes human environments (medical triage, crisis response, autonomous driving), "Attention Under Pressure" becomes a critical safety metric. 
If an emergency dispatch AI degrades in logical accuracy specifically because the caller expresses panic or fear, it poses a direct risk. Measuring and eliminating this "inherited human cognitive bias" is a necessary milestone on the path to robust General Intelligence.
