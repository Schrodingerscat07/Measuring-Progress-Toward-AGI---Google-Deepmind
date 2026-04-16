# CMAT: Cross-Modal Attention Triage
## Detailed Research Findings & Technical Analysis

The CMAT benchmark evaluates the working memory and "attentional triage" capabilities of frontier vision-language models (VLMs). Through testing 18 models across 500 samples, we uncovered a critical threshold where contemporary architectures fail.

### 1. The "I4 Cliff" — The Limits of Working Memory
The most significant finding of this research is the non-linear degradation of model accuracy against computational depth.

*   **Integration Levels 1 & 2 (Simple Lookup & Arithmetic):** Most modern frontier models (Gemini 2.5 Pro, GPT-5.4) and even mid-tier models (Gemma 3-27B) score 85%+ accuracy. The models successfully parse the text memo, locate the corresponding visual entity in the dashboard, and apply a 1-step mathematical transformation.
*   **Integration Level 3 (Conditionals):** A slight degradation occurs (avg -5% to -10%) when models must conditionally apply a correction based on a visual text field.
*   **Integration Level 4 (Weighted Aggregation):** This represents a "cognitive cliff." At this level, models must read the memo, filter out multiple entities from the visual dashboard, correct the values of the remaining entities based on textual notes, and compute an aggregate (e.g., weighted average).
    *   **Finding:** Accuracy drops precipitously. GPT-5.4 collapses from >90% to 24%. 
    *   **Conclusion:** The failure mode is not perceptual. The models successfully extract the numbers (as proven by I1/I2 scores) but cannot sustain the working memory required to hold multiple modified states across modalities before calculating a final result. This represents a core structural limitation in transformer attention windows when bridging modalities computationally.

### 2. Immunity to Adversarial Noise (Conflict Levels)
CMAT varies the "Conflict Level" (C1 to C5), introducing corrupted cells, contradictory preamble text, and visually dense "decoy summary panels."

*   **Finding:** Across all models, the delta between C1 (clean data) and C5 (maximum adversarial adversarial noise) was consistently **less than 4%**.
*   **Conclusion:** Modern instruction-tuned models have effectively "solved" basic visual distraction. They are highly capable of ignoring decoy information and adhering strictly to the primary task directive. The bottleneck is strictly computational, not distractibility.

### 3. The Gemma Generational Architecture Leap
By testing models across generations, CMAT provided a clear longitudinal view of architectural improvements.

*   **Gemma 3 (27B):** Scored 42% overall. It struggled significantly with cross-referencing, often falling for "image-only traps" (ignoring the text memo entirely).
*   **Gemma 4 (31B):** Scored 98.7% overall. This +56.7 percentage point increase is the largest generational delta observed.
*   **Conclusion:** The massive leap indicates that Gemma 4's cross-attention mechanism between vision encoders and text representation has fundamentally shifted from a "weak alignment" approach to a "unified representation" capable of deep relational reasoning.

### 4. Failure Mode Analysis: Why Models Fail
Every incorrect answer in CMAT is binned into two categories: "Image-trap" (the model calculated the answer using uncorrected visual data) or "Other Error" (hallucination or math failure).

*   **Finding:** For models above 60% accuracy, "Image-traps" constitute less than 15% of all errors. The vast majority of failures are arithmetic or sequential logic errors that occur *after* the correct multimodal data has been extracted.
*   **Implication for AGI:** To achieve generalized reasoning, scaling visual perception is no longer the key driver. Progress requires architectural innovations in latent reasoning (e.g., test-time compute, System 2 explicit reasoning tokens) to process extracted cross-modal data without losing state.
