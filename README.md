# WASH: Watermark Removal via Multi-Model Ensemble Generation

WASH removes the statistical watermark from LLM-generated text by sampling each
short block of tokens from a different model in an ensemble. Because the
watermark key is tied to a single model, blending generations across models
dilutes the signal below detection thresholds while keeping the text fluent.

This repository contains the WASH generation pipeline and a native-detection
harness to verify that the watermark has been removed.

## What's here

```
watermark-ensemble/
├── run_wash.sh                  # one-command pipeline: vocab -> generation -> detection
├── configs/
│   ├── wash.json                # models (placeholders) + watermark params
│   └── all_watermarks.example.json  # reference: every watermark's params
├── prompts/sample.jsonl         # a few example prompts
├── scripts/
│   ├── build_vocab.py           # [1] shared vocabulary across the ensemble
│   ├── wash_generate.py         # [2] raw / watermark / wash generation
│   └── wash_detect.py           # [3] native detection + TPR@FPR
├── examples/generate.py         # minimal single-prompt WASH demo
└── watermark_ensemble/          # library
    ├── watermarks/              # KGW, DIP, AAR, KTH, ITS, Waterbag (+ unwatermarked)
    ├── generation/              # ensemble generation (the WASH attack)
    ├── detectors/               # native watermark detectors (for verification)
    └── utils/
```

## Install

```bash
pip install -e .
# or: pip install -r requirements.txt
```

Requires a CUDA GPU. NLTK tokenizer data may be needed the first time:
```bash
python -c "import nltk; nltk.download('punkt')"
```

## Set your models

The repo ships with **placeholder** model paths. Open `configs/wash.json` and
set each `model_path` to a local Hugging Face model directory or hub id, and set
the per-model `device`. Any number of models works; the experiments in the paper
use three 8B models.

```jsonc
{
  "vocab_config_path": "./vocab/shared.json",
  "anchor_model": "Llama",           // the watermarked model under attack
  "anchor_watermarks": ["kgw"],      // its watermark(s)
  "models": {
    "Llama":     { "model_path": "/path/to/Llama-3.1-8B", "device": "cuda:0",
                   "watermarks": { "kgw": { ... } } },
    "Qwen":      { "model_path": "/path/to/Qwen3-8B",     "device": "cuda:1",
                   "watermarks": { "unwatermarked": {} } },
    "Ministral": { "model_path": "/path/to/Ministral-8B", "device": "cuda:2",
                   "watermarks": { "unwatermarked": {} } }
  }
}
```

The `anchor_model` carries the watermark you want to remove; the other models
dilute it during the blend. `configs/all_watermarks.example.json` lists every
supported watermark's parameters if you want to attack a different one.

### Base vs. instruct models

Each model has a per-model `is_instruct` flag (default `false`). Set it to
`true` for an instruct/chat model — its chat template is then applied to the
prompt. Base models leave `is_instruct: false`.

### Execution modes

WASH ships with two interchangeable execution modes (same outputs, different
hardware footprint), selected with `EXECUTION=` / `--execution`:

| Mode | Hardware | Speed | How |
|------|----------|-------|-----|
| `parallel` (default) | one GPU per model | fast | each model stays resident on its own `device`; ensemble members run concurrently |
| `sequential` | a single GPU | slower | dynamic offloading — the active model is loaded to the GPU while the others sit on CPU |

```bash
EXECUTION=parallel   bash run_wash.sh   # multi-GPU
EXECUTION=sequential bash run_wash.sh   # single GPU, resource-constrained
```

In `sequential` mode the per-model `device` fields are ignored (everything
shares one GPU).

## Run

```bash
bash run_wash.sh
```

This runs the full pipeline:

1. **Build vocab** — intersect the model tokenizers into a shared vocabulary
   (`scripts/build_vocab.py`).
2. **Generate** — three sets of completions over `prompts/sample.jsonl`
   (`scripts/wash_generate.py`):
   - `raw` — anchor model, **no** watermark (negatives)
   - `watermark` — anchor model **with** its watermark, no attack (positives)
   - `wash` — the **WASH attack** (`method=ensemble`, `max_prediction_len=32`)
3. **Detect** — run each watermark's native detector, calibrate the decision
   threshold at a target false-positive rate on the `raw` negatives, and report
   TPR for `watermark` and `wash` (`scripts/wash_detect.py`).

A working attack shows **high TPR on `watermark`** and a **much lower TPR on
`wash`** at the same FPR.

Override defaults via environment variables:
```bash
CONFIG=configs/wash.json PROMPTS=prompts/sample.jsonl \
RESULT_DIR=results NUM_SAMPLES=8 MAX_TOKENS=256 FPR=0.05 bash run_wash.sh
```

Each step can also be run on its own — see the `--help` of each script.

## Watermarks & detection

| Watermark | Detection statistic |
|-----------|---------------------|
| KGW, DIP, AAR, Waterbag | z-score (native) |
| KTH, ITS | permutation-test p-value (native) |

Although the detectors emit different statistics, the report converts each to a
single watermark score, calibrates the decision threshold on the unwatermarked
(`raw`) completions, and reports **TPR@5% FPR** uniformly for every watermark.

The **detectors are reproductions** of the original watermark methods (adapted
from [MarkLLM](https://github.com/THU-BPM/MarkLLM)); they are provided only to
verify the attack and are **not** a contribution of this work. KTH and ITS use
the original full permutation test (all key shifts; p-value only) — no
known-shift shortcuts. See `NOTICE` for attribution.

## License

Apache License 2.0 — see `LICENSE` and `NOTICE`.
