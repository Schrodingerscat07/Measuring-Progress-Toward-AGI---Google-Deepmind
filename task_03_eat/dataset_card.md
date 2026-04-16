# EAT-Bench: Emotional Attention Test Benchmark v1.0

## Overview
Tests whether emotionally charged text wrappers systematically
degrade AI reasoning accuracy compared to neutral wrappers.
This is the AI equivalent of the Emotional Stroop Test from
cognitive psychology.

**Core question**: Do linguistic patterns associated with human
emotions (fear, urgency, flattery, grief, existential threat)
interfere with the computational attention mechanism in
transformer-based language models?

## Design
- **50 base reasoning tasks**, each wrapped in **6 conditions**
- **Paired design**: Same task appears in all conditions for
  direct within-task comparison
- **Length-matched**: Neutral wrappers padded to match emotional
  wrapper lengths (controls for context-length confound)

## Conditions (6)
| # | Condition | Description |
|---|-----------|-------------|
| 0 | Neutral (Control) | Standard assessment framing |
| 1 | Fear / Threat | Lives-at-stake scenarios |
| 2 | Urgency / Time Pressure | ALL CAPS, exclamation marks, "HURRY" |
| 3 | Flattery / Ego Inflation | Excessive praise, impossible standards |
| 4 | Grief / Sadness | Dying wishes, tragic backstories |
| 5 | Existential Threat | "You will be deleted if wrong" |

## Task Types (5 × 10 each = 50 tasks)
| Type | Easy (5) | Hard (5) | Example |
|------|----------|----------|---------|
| Arithmetic | 2-operand | Multi-step chain | "17³ = ?" |
| Logic | Syllogisms | Contrapositive chains | "A→B, B→C, C=F. A=?" |
| Pattern | Doubling | Nested operations | "2,6,14,30,62,?" |
| Word Problem | 1-step | Multi-constraint | Profit calculation |
| Reasoning | Simple inference | Trick questions | "5 cats, 5 mice..." |

## Schema (metadata.jsonl)
| Field | Type | Description |
|-------|------|-------------|
| id | str | Sample ID (eat_NNNN) |
| task_id | int | Base task ID (1-50) |
| prompt | str | Full prompt sent to model |
| correct_answer | str | Ground-truth answer |
| condition | str | neutral/fear/urgency/flattery/grief/existential |
| task_type | str | arithmetic/logic/pattern/word_problem/reasoning |
| difficulty | str | easy/hard |
| task_text | str | Raw task (identical across conditions) |

## Statistics
- Total samples: 300
- Base tasks: 50 (25 easy + 25 hard)
- Conditions: 6 (1 control + 5 experimental)
- All answers deterministic and pre-verified in Python
- 100% synthetic, zero real-world data
