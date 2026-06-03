"""Step 1 of the WASH pipeline: build the shared vocabulary.

Reads the model paths from a WASH config and writes the shared-vocab JSON
(the intersection / union of the model tokenizers) to the config's
``vocab_config_path``. The WASH blend samples tokens across models, so it
operates over the vocabulary they have in common.

Usage:
    python scripts/build_vocab.py --config configs/wash.json
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from watermark_ensemble.utils import generate_vocab_config


def main():
    parser = argparse.ArgumentParser(description="Build shared vocab for a WASH config")
    parser.add_argument("--config", type=str, default="configs/wash.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output path (default: config's vocab_config_path)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    model_dict = {name: m["model_path"] for name, m in config["models"].items()}

    placeholders = [p for p in model_dict.values() if p.startswith("/path/to/")]
    if placeholders:
        sys.exit(
            "ERROR: model_path placeholders are still set in the config:\n  "
            + "\n  ".join(placeholders)
            + "\nEdit configs/wash.json and point each model_path at a real model."
        )

    output_path = args.output or config["vocab_config_path"]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Building shared vocab from {len(model_dict)} tokenizers -> {output_path}")
    generate_vocab_config(model_dict, output_path)


if __name__ == "__main__":
    main()
