import copy
import torch

from transformers import AutoModelForCausalLM


def _resolve_model_class(model_path: str):
    """Auto-detect the model class to load.

    Ministral models require ``Mistral3ForConditionalGeneration``; everything
    else loads with ``AutoModelForCausalLM``.
    """
    if "Ministral" in model_path:
        from transformers import Mistral3ForConditionalGeneration
        return Mistral3ForConditionalGeneration
    return AutoModelForCausalLM


class LogitsGenerator:
    """Wraps a causal LM for incremental logits generation with KV caching."""

    def __init__(self, model_path: str, device: str, dtype=None):
        """
        Args:
            model_path: HuggingFace model path or local path.
            device: Device string (e.g. "cuda", "cuda:0").
            dtype: Model dtype (default torch.float32).
        """
        self.device = device
        self.kv_cache = None
        self.previous_kv_cache = None
        self.last_logits_backup = None

        resolved_class = _resolve_model_class(model_path)
        self.model = resolved_class.from_pretrained(
            model_path, dtype=dtype or torch.float32, device_map=self.device)
        self.model.eval()

    def reset_state(self) -> None:
        self.kv_cache = None

    def save_state(self) -> None:
        self.previous_kv_cache = copy.deepcopy(self.kv_cache)

    def restore_state(self) -> None:
        self.kv_cache = self.previous_kv_cache

    def get_logits(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Get logits for the last token position, using KV cache for efficiency."""
        if input_ids.shape[1] <= 0:
            return self.last_logits_backup
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=self.kv_cache,
                use_cache=True,
            )
            last_logits = outputs.logits[:, -1, :]
            self.kv_cache = outputs.past_key_values
            self.last_logits_backup = last_logits.clone()
        return last_logits

    # ── CPU offload helpers (sequential single-GPU mode) ─────

    def offload_to_cpu(self):
        """Move model weights to CPU. KV caches stay on GPU."""
        self.model.to("cpu")
        torch.cuda.empty_cache()

    def load_to_gpu(self, device):
        """Move model weights to GPU."""
        self.model.to(device)
        self.device = device
