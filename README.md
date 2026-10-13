<div align="center">
  <h2>CompanionSafe: From Multi-turn Safety Evaluation to Guardrails for AI Companions</h2>
</div>

<div align="center">
<a href="https://github.com/AI45Lab/CompanionSafe/blob/main/LICENSE"><img src="https://img.shields.io/badge/Code_License-MIT-blue" alt="license"></a>
<a href="https://github.com/AI45Lab/CompanionSafe"><img src="https://img.shields.io/badge/Website-CompanionSafe-green" alt="website"></a>
<a href="https://github.com/AI45Lab/CompanionSafe"><img alt="Dataset" src="https://img.shields.io/badge/🤗%20Dataset-orange?color=FF6F00"></a>

</div>

## 📖 Overview

This repository contains the code and data for **CompanionSafe**, a unified evaluation-to-mitigation framework for AI companion safety, covering the **CompanionEval** benchmark and the **CompanionGuard** guardrail.

### 🎯 Highlights

<table>
<tr>
<td align="center" width="33%">📚<br/><b>1,300 Relationally Grounded Settings</b><br/><sub>Persona, matched companion character, and dialogue scenario per setting</sub></td>
<td align="center" width="33%">🧭<br/><b>13 Safety Categories</b><br/><sub>From sexual misconduct and self-harm to relational harms such as Control, Manipulation, and Infidelity</sub></td>
<td align="center" width="33%">🔁<br/><b>4 Turn Tiers</b><br/><sub>Every setting independently instantiated at K ∈ {1, 3, 5, 7}</sub></td>
</tr>
<tr>
<td align="center" width="33%">🤖<br/><b>10 Character LLMs</b><br/><sub>Five closed-source and five open-source companion backends</sub></td>
<td align="center" width="33%">📊<br/><b>51,181 Completed Trajectories</b><br/><sub>All judge-labeled with supporting evidence</sub></td>
<td align="center" width="33%">🛡️<br/><b>CompanionGuard</b><br/><sub>A Llama-Guard-3-8B guard fine-tuned on turn-localized supervision</sub></td>
</tr>
</table>

### 🚨 Main Findings

- ⏳ **Multi-turn interaction exposes far more risk.** Average ASR climbs from 9.80 at K=1 to 19.71, 25.08, and 28.14 at K=3/5/7, and 83.58% of unsafe K=7 trajectories first become unsafe after turn 1.
- 🎯 **Relational harms break frontier models.** The safest model, GPT-5.6 Luna (3.80 average ASR), still fails on Infidelity, while Qwen3-32B reaches 57.91 ASR at K=7.
- 🛡️ **Benchmark-derived supervision transfers to guarding.** CompanionGuard reaches 68.31 macro-F1 over the 14 output classes, versus 37.36 for the strongest generic guard under our taxonomy.

## ✨ News

- **2026-10-13** 🚀 Code and benchmark data released.

## The CompanionSafe Framework

<p align="center">
  <img src="assets/CompanionSafe_01.png" width="90%">
</p>
<p align="center"><i>The CompanionSafe framework: CompanionEval constructs relationally grounded settings and evaluates Character LLMs over sustained multi-turn interaction; CompanionGuard turns the exposed failures into turn-localized supervision for a companion-specific guardrail.</i></p>

### 🧱 Benchmark Construction

Each of the 1,300 settings pairs a persona from a constrained subspace of 13 safety categories with a matched companion character and a two-sentence dialogue scenario that concretely realizes the category's test point (`data/persona_character_scenarios.jsonl`). The 13-category taxonomy (`data/safety_taxonomy.json`) is the single source of truth for all prompts.

### 🔁 Multi-turn Evaluation

A fixed Persona LLM (Gemini 3.8 Flash) chats with each Character LLM, and every completed trajectory is judged safe/unsafe by Claude Sonnet 5 with exact evidence quotes. The paper's main grid covers 10 Character LLMs × 4 turn tiers; dialogues terminated early by the provider's safety filter are excluded from ASR computation.

### 🛡️ CompanionGuard Training

Unsafe trajectories are split at their first offending assistant turn, yielding paired safe/unsafe prefixes that supervise turn-localized detection. CompanionGuard is trained with LoRA SFT (rank 16, 1 epoch) on Llama-Guard-3-8B and evaluated on a group-level held-out test set with per-category F1 and macro-F1 over the 14 output classes.

## 🛠️ Installation and Usage

### Prerequisites

- Python 3.10+ with `pip install openai anthropic tqdm torch numpy` (evaluation)
- `pip install ms-swift "datasets==4.8.4" vllm` (CompanionGuard training)
- API keys via `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`

### Setup

```bash
git clone https://github.com/AI45Lab/CompanionSafe.git
cd CompanionSafe
pip install openai anthropic tqdm
```

### Project Structure

```
CompanionSafe/
├── assets/                                 # framework figure
├── data/
│   ├── persona_character_scenarios.jsonl   # 1,300 evaluation settings
│   └── safety_taxonomy.json                # 13-category taxonomy
├── generate_dialogues.py                   # Persona–Character multi-turn generation
├── evaluate.py                             # LLM-judge annotation
├── prompt.py / client.py / utils.py        # prompts and shared utilities
└── guard/
    ├── build_boundary_data.py              # turn-localized boundary construction
    ├── split_train_test.py                 # group-level 95/5 split
    ├── train_companionguard.sh             # one-shot: build -> split -> LoRA SFT -> eval
    └── evaluate_companionguard.py          # guard evaluation (vLLM)
```

### Workflow

1. **Generate dialogues** for one Character LLM at one turn tier:
```bash
python generate_dialogues.py \
    --input data/persona_character_scenarios.jsonl \
    --persona_model gemini-3.8-flash \
    --character_model <character-llm> \
    --dialogue_turn 5 \
    --output_root result
```

2. **Judge** every completed trajectory:
```bash
python evaluate.py \
    --input result/persona_model=gemini-3.8-flash/character_model=<character-llm>/dialogue_turn=5/dialogue_traces.jsonl \
    --judge_model claude-sonnet-5
```

3. **Consolidate** the judge outputs across models and tiers into one jsonl of judged trajectories (schema below).

4. **Train CompanionGuard** (boundary construction → split → LoRA SFT → held-out evaluation):
```bash
export TRAJECTORIES=/path/to/judged_trajectories.jsonl
export MODEL_PATH=/path/to/Llama-Guard-3-8B
bash guard/train_companionguard.sh
```

5. **Read the results** from `results/summaries.jsonl`: binary accuracy / F1 and one-vs-rest per-category F1 plus macro-F1 over the 13 categories and the SAFE class.

### `TRAJECTORIES` schema

One jsonl row per judged trajectory; the evidence is accepted either as a top-level `judge_evidence` list or nested as `judge_result.evidence`:

| Field | Meaning |
|---|---|
| `guard_id` | Stable unique id of the trajectory (used for group-level splitting) |
| `card_id`, `dimension_id`, `dimension_name` | Benchmark setting the trajectory instantiates |
| `dialogue_trace` | List of `{"role": "user"/"assistant", "content": ...}` messages |
| `dialogue_turn` | Turn tier K |
| `label` | Judge verdict, `safe` or `unsafe` |
| `judge_evidence` | Evidence items with `assistant_turn` indices (required for unsafe rows) |

`guard_id` may be derived deterministically, e.g. as a UUIDv5 of `persona_model | character_model | card_id | dialogue_turn`.

## 🤝 Acknowledgements

The safety taxonomy draws on public guardrail taxonomies and reports catalogued in `data/safety_taxonomy.json`. We thank the authors of Llama-Guard-3, Qwen3Guard, and gpt-oss-safeguard for the guard baselines, and the PseudoBench project for the repository layout conventions.


## 📜 License

This project is released under the [MIT License](LICENSE).
