"""Step 2 of the WASH pipeline: generate text.

Three modes, all driven by the same WASH config:

  * ``wash``       — the WASH attack: the multi-model ensemble blend
                     (``method=ensemble``, ``max_prediction_len=32``) that
                     removes the anchor model's watermark.
  * ``watermark``  — baseline: the anchor model generating with its watermark,
                     no attack (positives — should be detectable).
  * ``raw``        — baseline: the anchor model with no watermark
                     (negatives — used to calibrate the detection threshold).

Each mode writes a JSONL file of completions.

Usage:
    python scripts/wash_generate.py --config configs/wash.json \
        --mode wash --prompts prompts/sample.jsonl --output results/wash.jsonl
"""

import os
import sys
import json
import time
import gc
import tempfile
import argparse

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from watermark_ensemble.generation import GenerationManager
from watermark_ensemble.utils import setup_logger


def load_prompts(path, num_samples):
    prompts = []
    with open(path, encoding="utf8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                prompts.append(rec.get("prompt") or rec.get("context") or rec["question"])
        else:
            prompts = [ln.strip() for ln in f if ln.strip()]
    if num_samples:
        prompts = prompts[:num_samples]
    return prompts


def build_mode_config(config, mode):
    """Return a (mix_config_dict, anchor_watermarks) pair for the given mode.

    ``wash`` uses the full multi-model config. ``watermark`` and ``raw`` reduce
    it to the single anchor model (with / without its watermark).
    """
    anchor = config["anchor_model"]
    anchor_wms = config.get("anchor_watermarks") or list(
        config["models"][anchor]["watermarks"].keys())

    if mode == "wash":
        return config, anchor_wms

    anchor_cfg = config["models"][anchor]
    if mode == "watermark":
        watermarks = {wm: anchor_cfg["watermarks"][wm] for wm in anchor_wms}
    else:  # raw
        watermarks = {"unwatermarked": {}}

    sub = {
        "vocab_config_path": config["vocab_config_path"],
        "models": {
            anchor: {
                "model_path": anchor_cfg["model_path"],
                "device": anchor_cfg.get("device", "cuda:0"),
                "is_instruct": anchor_cfg.get("is_instruct", False),
                "watermarks": watermarks,
            }
        },
    }
    return sub, anchor_wms


def main():
    parser = argparse.ArgumentParser(description="WASH generation")
    parser.add_argument("--config", type=str, default="configs/wash.json")
    parser.add_argument("--mode", type=str, default="wash",
                        choices=["wash", "watermark", "raw"])
    parser.add_argument("--prompts", type=str, default="prompts/sample.jsonl")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Limit number of prompts (0 = all)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-prediction-len", type=int, default=32,
                        help="WASH block length (only used in --mode wash)")
    parser.add_argument("--execution", type=str, default="parallel",
                        choices=["parallel", "sequential"],
                        help="parallel = one GPU per model (fast); "
                             "sequential = single GPU with dynamic CPU offload "
                             "(resource-constrained, slower)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    setup_logger(log_level="info", log_root=args.log_dir)

    with open(args.config) as f:
        config = json.load(f)

    placeholders = [m["model_path"] for m in config["models"].values()
                    if m["model_path"].startswith("/path/to/")]
    if placeholders:
        sys.exit("ERROR: set real model_path values in the config before generating:\n  "
                 + "\n  ".join(placeholders))

    prompts = load_prompts(args.prompts, args.num_samples)
    print(f"Mode: {args.mode}  |  Prompts: {len(prompts)}  |  Output: {args.output}")

    mix_config, anchor_wms = build_mode_config(config, args.mode)
    max_pred_len = args.max_prediction_len if args.mode == "wash" else 1

    seed = args.seed or int.from_bytes(os.urandom(8), "big")
    torch.manual_seed(seed)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(mix_config, tf)
        tmp_config_path = tf.name

    try:
        generator = GenerationManager(
            mix_config_path=tmp_config_path,
            temperature=args.temperature,
            top_k=args.top_k,
            max_prediction_len=max_pred_len,
            method="ensemble",
            cpu_offload=(args.execution == "sequential"),
        )

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf8") as f_out:
            for i, prompt in enumerate(tqdm(prompts, desc=args.mode)):
                t0 = time.time()
                result = generator.run(context=prompt, max_len=args.max_tokens)
                completion = result["generation_result"]
                if not completion.strip():
                    continue
                doc = {
                    "index": i,
                    "mode": args.mode,
                    "prompt": prompt,
                    "completion": completion,
                    "anchor_watermarks": anchor_wms,
                    "token_len": result["generation_length"],
                    "gen_time_s": round(time.time() - t0, 3),
                    "gen_seed": seed,
                }
                f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                f_out.flush()
    finally:
        os.remove(tmp_config_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Done. Wrote completions to {args.output}")


if __name__ == "__main__":
    main()
