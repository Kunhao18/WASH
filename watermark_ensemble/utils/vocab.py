import json

from tqdm import tqdm
from transformers import AutoTokenizer


def generate_vocab_config(model_dict: dict, output_path: str) -> dict:
    """Generate a shared vocabulary config from multiple model tokenizers.

    Args:
        model_dict: Mapping of model names to model paths.
            e.g. {"Llama": "/path/to/Llama", "Qwen": "/path/to/Qwen"}
        output_path: Path to save the resulting JSON config.

    Returns:
        The vocab config dict with keys: tokenizers, common_vocab, union_vocab.
    """
    tokenizer_list = []
    for model_path in tqdm(model_dict.values(), desc="Loading tokenizers"):
        tokenizer_list.append(
            AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        )

    common_keys = None
    union_keys = set()

    for tokenizer in tqdm(tokenizer_list, desc="Processing vocabs"):
        vocab_keys = set(tokenizer.vocab.keys())
        union_keys |= vocab_keys
        common_keys = vocab_keys if common_keys is None else common_keys & vocab_keys
        if tokenizer.eos_token is not None:
            common_keys.discard(tokenizer.eos_token)

    common_vocab = sorted(list(common_keys))
    union_vocab = sorted(list(union_keys))

    print(
        f"Common vocab size: {len(common_vocab)}\n"
        f"Union vocab size: {len(union_vocab)}"
    )

    vocab_config = {
        "tokenizers": model_dict,
        "common_vocab": common_vocab,
        "union_vocab": union_vocab,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_config, f, indent=2, ensure_ascii=False)

    print(f"Saved vocab config to {output_path}")
    return vocab_config
