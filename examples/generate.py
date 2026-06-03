"""Minimal example: run the WASH ensemble on a single prompt.

    python examples/generate.py --config configs/wash.json \
        --prompt "Once upon a time"
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from watermark_ensemble.generation import GenerationManager
from watermark_ensemble.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="WASH ensemble generation (single prompt)")
    parser.add_argument("--config", type=str, default="configs/wash.json")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-prediction-len", type=int, default=32,
                        help="WASH block length")
    parser.add_argument("--execution", type=str, default="parallel",
                        choices=["parallel", "sequential"],
                        help="parallel = one GPU per model; "
                             "sequential = single GPU with dynamic offload")
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    setup_logger(log_level="info", log_root=args.log_dir)

    generator = GenerationManager(
        mix_config_path=args.config,
        temperature=args.temperature,
        top_k=args.top_k,
        max_prediction_len=args.max_prediction_len,
        method="ensemble",
        cpu_offload=(args.execution == "sequential"),
    )

    result = generator.run(context=args.prompt, max_len=args.max_len)

    print("=== Prompt ===")
    print(args.prompt)
    print("=== Generation ===")
    print(result["generation_result"])
    print(f"\nLength: {result['generation_length']}, "
          f"Time: {result['generation_time']:.2f}s")


if __name__ == "__main__":
    main()
