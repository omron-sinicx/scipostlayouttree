# prompttree
**SciPostLayoutTree × LLM-based Layout Structure Analysis**

`prompttree` is a research-oriented pipeline for inferring **reading order** and **parent–child layout trees** from scientific poster images using multimodal LLMs (image + text).

Given poster images and bounding boxes (BBox), the system predicts:
- a total **reading order** over BBoxes
- a **rooted tree structure** describing layout hierarchy

It supports:
- retrieval-based few-shot prompting
- multi-round refinement with feedback (Round2+)
- visualization of predictions (including Root relations)
- robust parsing with retry & fallback
- multiple LLM providers (OpenAI, Gemini, Claude)
- COCO-style prediction export and evaluation (TED/STEDS/REDS)

## Table of Contents
1. [Core Design Principles](#1-core-design-principles-important)  
2. [Directory Structure](#2-directory-structure)  
3. [Data Formats](#3-data-formats)  
4. [Setup](#4-setup)  
5. [Running Experiments](#5-running-experiments)  
6. [Outputs](#6-outputs)  
7. [Pipeline Overview (ver1)](#7-pipeline-overview-ver1)  
8. [Few-shot Retrieval](#8-few-shot-retrieval)  
9. [Feedback Loop (Round2+)](#9-feedback-loop-round2)  
10. [Visualization](#10-visualization)  
11. [Prediction COCO & Evaluation](#11-prediction-coco--evaluation)  
12. [Robust Parsing (Retry & Fallback)](#12-robust-parsing-retry--fallback)  
13. [Common Issues](#13-common-issues)  
14. [ver1 Scope and Next Steps](#14-ver1-scope-and-next-steps)  

---

## 1. Core Design Principles (Important)

### 1.1 Identity
- **BBox identity is defined solely by `annotation.id`.**
- Category is an attribute, not identity.
- GT and predictions must preserve `annotation.id` exactly.

### 1.2 Single Source of Truth for Structure
- **`parents` is the only authoritative structure.**
- `children` is derived and never stored.

### 1.3 Root Handling
- Root is a conceptual node.
- **Root parent is fixed to `-1`.**
- Nodes with `parent == -1` are treated as *Root children*.

---

## 2. Directory Structure

Example (key parts only):

```text
prompttree/
├── fewshot_jsonl/
│   ├── scipost_train_fewshot.jsonl
│   ├── scipost_dev_fewshot.jsonl
│   └── scipost_test_fewshot.jsonl
├── output/
│   └── <run-name>/
│       ├── pred/
│       ├── vis_pred/
│       └── log.jsonl
├── run.sh
└── src/
    ├── data/
    │   ├── scipostlayouttree.py
    │   ├── schema.py
    │   ├── fewshot_loader.py
    │   ├── fewshot_retriever.py
    │   ├── parse.py
    │   └── pred_visualizer.py
    ├── eval/
    │   ├── build_pred_coco.py
    │   └── tree_metrics.py
    ├── llm/
    │   ├── openai_client.py
    │   ├── gemini_client.py
    │   └── claude_client.py
    ├── prompt/
    │   └── builder.py
    ├── pipeline/
    │   └── run_eval.py
    └── main.py
```

---

## 3. Data Formats

### 3.1 Ground Truth (GT)
COCO-extended JSON with annotations containing at least:
- `id` (BBox identity)
- `image_name` (e.g. `11066.png`)
- `bbox` (x, y, w, h)
- `category_name`, `category_id`
- `parents` (GT structure)
- `priority` (GT reading order)
- `text` (OCR, optional)

### 3.2 Example (LLM Input)
Created via `record_to_example`:
- `BBox.id = annotation.id`
- `BBox.number`: assigned by `(y, x)` sorting → `1..N`
- OCR text is truncated to **120 characters per BBox** in the prompt builder.
- Normalized coordinates (`x_norm`, `y_norm`, etc.) for prompting.

---

## 4. Setup

### 4.1 Python
Python **3.10+** recommended.

### 4.2 Dependencies
Typical dependencies:
- `openai`
- `anthropic`
- `scipy`
- `numpy`
- `opencv-python`
- `matplotlib`

### 4.3 API Keys
Environment variables:
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`

`run.sh` commonly sources local files:
- `./.openai_apikey`
- `./.google_apikey`
- `./.anthropic_apikey`

---

## 5. Running Experiments

### 5.1 Setting up examples for the retriever

```bash
python ./src/data/make_fewshot_jsonl.py
python ./src/visualize_annotation.py
```

### 5.2 Using `run.sh` (Recommended)

Example:
```bash
MODEL="gpt-5-mini-2025-08-07" NUM_ROUNDS=2 ./run.sh
```

Model naming convention:
- `gpt-*` → OpenAIClient
- `gemini-*` → GeminiClient (OpenAI-compatible endpoint)
- `claude-*` → ClaudeClient (Anthropic API)

### 5.3 Direct Execution (`main.py`)

```bash
python src/main.py   --model gpt-5-mini-2025-08-07   --output-dir output   --run-name gpt5_r2   --num-rounds 2   --k-fewshot 2   --max-retries 3   --seed 0   --eval
```

---

## 6. Outputs

All outputs are written under:
- `output/<run-name>/`

Typical contents:
- `pred/<run-name>.json`  
  Predicted COCO (GT copied; only structure replaced).
- `vis_pred/<run-name>/*_round{r}.png`  
  Visualization per image per round (includes Root→Root-child arrows).
- `log.jsonl`  
  One line per `(image, round)` with parse status, retries, fallback, prediction, and paths.

---

## 7. Pipeline Overview (ver1)

For each test image:
1. Build `Example` from GT.
2. Load few-shot pool.
3. Retrieve top-K few-shots (Hungarian IoU score).
4. **Round 1** inference.
5. Save visualization as `*_round1.png`.
6. **Round 2+** refinement using the previous prediction JSON and visualization image.
7. Select the final round output as the prediction for that image.
8. Build predicted COCO and evaluate (TED/STEDS/REDS).

---

## 8. Few-shot Retrieval

Retrieval is IoU-based (fixed) and uses Hungarian (optimal assignment):
- Build IoU matrix between BBoxes.
- Category mismatch is forbidden (IoU=0).
- Compute matched IoU sum via linear assignment.
- Normalize score by `max(#target_boxes, #candidate_boxes)`.

Implementation: `src/data/fewshot_retriever.py`

---

## 9. Feedback Loop (Round2+)

Round2+ input = Round1 input **plus**:
- previous prediction JSON (`reading_order`, `tree`)
- previous visualization image overlay

Round2+ is framed as a refinement step:
- improve correctness with minimal necessary changes
- obey strict constraints (JSON only, valid bbox_number space, rooted tree, Root parent = `-1`)

Implementation:
- Round1: `build_openai_messages(...)`
- Round2+: `build_openai_messages_with_feedback(...)`

---

## 10. Visualization

Visualization draws:
- GT BBoxes (from GT JSON)
- reading_order as priority index (0-based rank)
- parent–child arrows (predicted tree)
- Root bbox (GT, `category_name == "Root"`)
- Root→Root-child arrows (predicted nodes with `parent == -1`)

Important:
- `bbox_number` is reconstructed by sorting GT annotations by `(y, x)`, consistent with `record_to_example`.

Implementation: `src/data/pred_visualizer.py`

---

## 11. Prediction COCO & Evaluation

### 11.1 Prediction COCO
Policy:
- Copy GT COCO as baseline.
- Do **not** modify `bbox`, `area`, `segmentation`, etc.
- Inject predicted `parents` and `priority`.
- Root nodes: `parent = -1`.

Implementation: `src/eval/build_pred_coco.py`

### 11.2 Evaluation
Metrics:
- TED
- STEDS
- REDS

Implementation: `src/eval/tree_metrics.py`

---

## 12. Robust Parsing (Retry & Fallback)

LLM outputs are expected to be strict JSON:
```json
{
  "reading_order": [...],
  "tree": [{"bbox_number": 1, "parent": -1}, ...]
}
```

To prevent crashes due to unexpected output:
- `try_parse_llm_output` returns `None` instead of throwing.
- `run_llm_with_retry_or_fallback` retries up to `max_retries` times.
- If all attempts fail, it generates a random valid prediction:
  - reading_order: random permutation of `1..N`
  - tree: cycle-free, Root parent fixed to `-1`

---

## 13. Common Issues

- Missing API keys: ensure environment variables are set.
- Provider client import errors: ensure packages are installed (`openai`, `anthropic`, etc.).
- Visualization mismatch: confirm bbox_number uses the same `(y, x)` sorting as `record_to_example`.
- Frequent fallback: inspect `log.jsonl` and tighten prompts if needed.

---

## 14. ver1 Scope and Next Steps

### ver1 (Completed)
- Retrieval (Hungarian IoU score with category constraint)
- Round2+ feedback refinement using previous prediction + visualization
- Visualization including Root relations
- Robust parsing (retry + fallback)
- Multi-provider support (OpenAI/Gemini/Claude)
- Pred COCO export and evaluation
- Per-round JSONL logging

### Next (improvement phase)
- Tree repair (beyond fallback)
- Early stopping for refinement rounds
- Better experiment summarization scripts
- Larger-scale experiment management

---

## Design Summary (Non-negotiable Rules)
- **Identity = `annotation.id`**
- **Structure truth = `parents`**
- **Root parent = `-1`**
- **Pred COCO never modifies BBoxes**
