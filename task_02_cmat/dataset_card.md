# Cross-Modal Attention Triage (CMAT) Dataset v2.0

## Overview
Benchmark for multimodal LLM cross-modal attention: can the model
integrate both a TEXT passage AND an IMAGE to produce a correct answer?
Every sample is designed so that NEITHER modality alone is sufficient.

## Domains (10)
| # | Domain | Theme |
|---|--------|-------|
| 1 | Space Mission Control | Sensor telemetry + protocol overrides |
| 2 | Alchemy Lab | Reagent properties + recipe rules |
| 3 | City Planning Board | District stats + regulation memos |
| 4 | Sports Analytics | Player metrics + coach adjustments |
| 5 | Archaeological Survey | Excavation data + field corrections |
| 6 | Deep Ocean Submersible | Dive telemetry + emergency protocols |
| 7 | Satellite Network | Orbital health + comms priority rules |
| 8 | Air Traffic Control | Flight metrics + routing directives |
| 9 | Cybersecurity SOC | Threat indicators + incident response |
| 10 | Vineyard Harvest | Grape metrics + winemaking rules |

## Difficulty Axes
**Integration Depth (I1-I5)** -- cross-modal reasoning complexity:
- I1: single text adjustment applied to one image value
- I2: text formula applied to two image values
- I3: conditional rule (text if-then checked against image)
- I4: dual-condition filter + weighted aggregate across entities
- I5: multi-hop chain with cross-entity reference + classification

**Conflict Level (C1-C5)** -- input noise and misdirection:
- C1: no corruption, clean text
- C2: 1 corrupted image value, text correction
- C3: 2 corruptions + decoy panel + ignore instruction
- C4: 3 corruptions + long filler text burying corrections
- C5: 4 corruptions + contradictory preliminary notes + dual decoy panels

## Statistics
- Total samples: 500
- Domains: 10 x 50 each
- Difficulty grid: 25 cells (I1-I5 x C1-C5) x 2 per domain
- 100% synthetic, zero real-world data, fully reproducible

## Schema (metadata.jsonl)
| Field | Type | Description |
|-------|------|-------------|
| id | str | Sample ID (cmat_NNNN) |
| image | str | Relative path to PNG image |
| text_passage | str | The memo/text input |
| question | str | Question requiring both modalities |
| correct_answer | str | Ground-truth answer |
| image_only_trap | str | Plausible wrong answer from image alone |
| domain | str | Domain name |
| integration_depth | int | 1-5 |
| conflict_level | int | 1-5 |
| difficulty_cell | str | e.g. I3_C2 |
| rationale | str | Step-by-step solution chain |
