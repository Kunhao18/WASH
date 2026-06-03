"""Step 3 of the WASH pipeline: native watermark detection + TPR@FPR.

Runs each watermark's own native detector (the same detector the original
authors published) on the generated completions, then reports detection power.
Because the detectors emit different statistics (z-score for KGW/AAR/DIP/
Waterbag, permutation p-value for KTH/ITS), they are mapped to a single
"watermark score" (higher = more watermarked) and the decision threshold is
calibrated on the raw (unwatermarked) completions at a target false-positive
rate. TPR is the fraction of watermarked / WASH completions above that
threshold.

A successful WASH attack shows high TPR on ``watermark`` completions and a
sharply lower TPR on ``wash`` completions.

Usage:
    python scripts/wash_detect.py --config configs/wash.json \
        --inputs results/raw.jsonl results/watermark.jsonl results/wash.jsonl
"""

import os
import sys
import json
import math
import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from watermark_ensemble.detectors import DETECTOR_MAP
from watermark_ensemble.utils import setup_logger

# Detectors whose score is a permutation p-value (lower = more watermarked).
PVALUE_SCHEMES = ("kth", "its")


def create_detector(scheme, watermark_config, vocab_size, device):
    return DETECTOR_MAP[scheme](
        temperature=1.0, top_k=0, vocab_size=vocab_size,
        watermark_type=scheme, config=watermark_config, device=device,
    )


def detection_score(scheme, detector, tokenizer, prompt, completion,
                    n_runs, max_detect_tokens):
    """Run the native detector and return a score where higher = watermarked."""
    prompt_ids = tokenizer([prompt], return_tensors="pt",
                           add_special_tokens=True)["input_ids"][0]
    completion_ids = tokenizer([completion], return_tensors="pt",
                               add_special_tokens=False)["input_ids"][0]
    if max_detect_tokens is not None and len(completion_ids) > max_detect_tokens:
        completion_ids = completion_ids[:max_detect_tokens]
    prefix_len = getattr(detector, "prefix_length", 4)
    check_ids = torch.cat([prompt_ids[-prefix_len:], completion_ids], dim=0)

    if scheme in PVALUE_SCHEMES:
        result = detector.detect(check_ids, n_runs=n_runs)
        p = float(result["p_value"])
        return -math.log10(max(p, 1e-12))  # higher = more watermarked
    else:
        result = detector.detect(tokenized_text=check_ids)
        return float(result["z_score"])


def main():
    parser = argparse.ArgumentParser(description="Native watermark detection + TPR@FPR")
    parser.add_argument("--config", type=str, default="configs/wash.json")
    parser.add_argument("--inputs", type=str, nargs="+", required=True,
                        help="JSONL files from wash_generate.py (include the raw file)")
    parser.add_argument("--watermarks", type=str, nargs="+", default=None,
                        help="Watermarks to detect (default: config anchor_watermarks)")
    parser.add_argument("--fpr", type=float, default=0.05,
                        help="Target false-positive rate for threshold calibration")
    parser.add_argument("--n-runs", type=int, default=100,
                        help="Permutation runs for KTH/ITS detection")
    parser.add_argument("--max-detect-tokens", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    setup_logger(log_level="info", log_root=args.log_dir)

    with open(args.config) as f:
        config = json.load(f)
    anchor = config["anchor_model"]
    anchor_cfg = config["models"][anchor]
    watermarks = args.watermarks or config.get("anchor_watermarks") \
        or list(anchor_cfg["watermarks"].keys())

    tokenizer = AutoTokenizer.from_pretrained(anchor_cfg["model_path"],
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    detectors = {}
    for wm in watermarks:
        if wm not in anchor_cfg["watermarks"]:
            print(f"WARNING: {wm} not in anchor '{anchor}' config, skipping")
            continue
        detectors[wm] = create_detector(wm, anchor_cfg["watermarks"][wm],
                                         len(tokenizer), args.device)
    print(f"Detecting {list(detectors.keys())} with anchor '{anchor}'\n")

    # scores[label][wm] = list of per-sample scores. label = file's generation mode.
    scores = {}
    for path in args.inputs:
        with open(path) as f:
            records = [json.loads(l) for l in f if l.strip()]
        label = records[0].get("mode", os.path.basename(path)) if records else path
        scores.setdefault(label, {wm: [] for wm in detectors})
        for rec in tqdm(records, desc=f"detect {label}"):
            completion = rec.get("completion", "")
            prompt = rec.get("prompt", rec.get("question", ""))
            if not completion.strip():
                continue
            for wm, det in detectors.items():
                try:
                    s = detection_score(wm, det, tokenizer, prompt, completion,
                                        args.n_runs, args.max_detect_tokens)
                except Exception as e:
                    print(f"  [{wm}] error: {e}")
                    continue
                scores[label][wm].append(s)

    # Calibrate threshold on the raw (unwatermarked) negatives, report TPR.
    if "raw" not in scores:
        print("\nNo 'raw' completions found in inputs — cannot calibrate FPR. "
              "Reporting mean scores only.")
    print(f"\n{'='*64}")
    print(f"Detection summary  (threshold calibrated at FPR={args.fpr:.0%} on raw)")
    print(f"{'='*64}")
    for wm in detectors:
        neg = scores.get("raw", {}).get(wm, [])
        thresh = (np.quantile(neg, 1.0 - args.fpr) if neg else float("nan"))
        print(f"\n[{wm}]  raw negatives: {len(neg)}   threshold: {thresh:.3f}")
        for label in scores:
            vals = scores[label][wm]
            if not vals:
                continue
            mean_s = float(np.mean(vals))
            if neg:
                tpr = float(np.mean(np.array(vals) > thresh))
                tag = "FPR" if label == "raw" else "TPR"
                print(f"    {label:<10} n={len(vals):<4} mean_score={mean_s:7.3f}   "
                      f"{tag}@{args.fpr:.0%}={tpr:.3f}")
            else:
                print(f"    {label:<10} n={len(vals):<4} mean_score={mean_s:7.3f}")


if __name__ == "__main__":
    main()
