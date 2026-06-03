import torch

from transformers import AutoTokenizer

from ..watermarks import WATERMARK_MAP
from ..utils.word_check import keep_last_word
from ..utils.logging import file_logger
from .logits_generator import LogitsGenerator


class ModelManager:
    """Manages a single model with its watermarks for generation tasks."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        max_prediction_len: int = 32,
        device: str = "cuda",
        is_instruct: bool = False,
        dtype=None,
    ):
        self.device = device
        self.model_name = model_name
        self.is_instruct = is_instruct

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = len(self.tokenizer)

        # Build stop token IDs set (for instruct models with multiple stop tokens)
        self.stop_token_ids = {self.tokenizer.eos_token_id}
        if is_instruct and hasattr(self.tokenizer, "added_tokens_encoder"):
            for tok_str, tok_id in self.tokenizer.added_tokens_encoder.items():
                if any(kw in tok_str.lower() for kw in ["eot", "end_of_turn", "endoftext", "<|end|>"]):
                    self.stop_token_ids.add(tok_id)

        self.watermarks = {}
        self.watermark_index = {}
        self.watermark_weights = None

        self.max_prediction_len = max_prediction_len
        self.do_resync = max_prediction_len > 1

        self.logits_generator = LogitsGenerator(
            model_path=model_path, device=device, dtype=dtype,
        )
        self.ctx_ids_cache = None
        self.ctx_ids_cache_backup = None
        self.watermark_choice_backup = None
        self._kv_len = 0  # number of tokens currently in KV cache

        # Resync tracking
        self._step_count = 0
        self._resync_count = 0
        self._resync_lengths = []

    @property
    def vocab(self):
        return self.tokenizer.vocab.keys()

    @property
    def num_watermarks(self):
        return len(self.watermarks)

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    def setup_vocab(self, common_vocab: list) -> None:
        self.reserve_ids = self.tokenizer.convert_tokens_to_ids(common_vocab)
        for stop_id in self.stop_token_ids:
            if stop_id not in self.reserve_ids:
                self.reserve_ids.append(stop_id)
        self.reserve_ids = torch.tensor(self.reserve_ids, dtype=torch.long, device=self.device)

    def setup_watermarks(self, watermarks: dict, temperature: float, top_k: int) -> None:
        for idx, (watermark_name, watermark_config) in enumerate(watermarks.items()):
            current_watermark = WATERMARK_MAP[watermark_name](
                temperature=temperature,
                top_k=top_k,
                vocab_size=self.vocab_size,
                watermark_type=watermark_name,
                config=watermark_config,
                device=self.device,
            )
            self.watermarks[watermark_name] = current_watermark
            self.watermark_index[idx] = current_watermark
        self.watermark_weights = torch.ones(len(watermarks), dtype=torch.float)

    def _save_state(self):
        self.ctx_ids_cache_backup = self.ctx_ids_cache.clone()
        self._kv_len_backup = self._kv_len
        self.logits_generator.save_state()
        for watermark in self.watermarks.values():
            watermark.save_state()

    def _restore_state(self):
        self.ctx_ids_cache = self.ctx_ids_cache_backup
        self._kv_len = self._kv_len_backup
        self.logits_generator.restore_state()
        for watermark in self.watermarks.values():
            watermark.restore_state()

    def _get_watermark_prob(self, logits, use_backup=False):
        if use_backup:
            watermark_choice = self.watermark_choice_backup
        else:
            watermark_choice = torch.multinomial(
                self.watermark_weights, num_samples=1, replacement=True)[0]
            self.watermark_choice_backup = watermark_choice
        modified_prob = self.watermark_index[int(watermark_choice)].apply_watermark(
            logits=logits, input_ids=self.ctx_ids_cache
        )
        return modified_prob

    def _get_prediction_seq(self, new_ids):
        prediction_len = 1
        max_prediction_len = min(4, self.max_prediction_len)
        hit_stop = False
        self.ctx_ids_cache = torch.cat([self.ctx_ids_cache, new_ids], dim=1)
        for watermark in self.watermark_index.values():
            watermark.update_params(step=1)

        while max_prediction_len <= self.max_prediction_len:
            while prediction_len < max_prediction_len:
                logits = self.logits_generator.get_logits(input_ids=new_ids)
                logits = logits[:, :self.vocab_size]
                watermarked_prob = self._get_watermark_prob(logits)
                new_ids = torch.multinomial(watermarked_prob, num_samples=1)

                # Stop predicting if we hit any stop token
                if new_ids[0, 0].item() in self.stop_token_ids:
                    hit_stop = True
                    break

                self.ctx_ids_cache = torch.cat([self.ctx_ids_cache, new_ids], dim=1)
                for watermark in self.watermark_index.values():
                    watermark.update_params(step=1)
                prediction_len += 1

            new_raw_ids = self.ctx_ids_cache[0, -prediction_len:]
            result_dict = keep_last_word(self.tokenizer, new_raw_ids,
                                         stop_token_ids=self.stop_token_ids)

            if hit_stop:
                # If we hit a stop token, return whatever word we have so far
                if result_dict["new_str"]:
                    result_dict["terminate"] = True
                    result_dict["prediction_len"] = prediction_len
                    return result_dict
                # No word extracted yet — decode what we have
                current_str = self.tokenizer.decode(new_raw_ids, skip_special_tokens=True)
                return {"new_str": current_str, "terminate": True,
                        "prediction_len": prediction_len}

            if result_dict["terminate"] or result_dict["new_str"]:
                result_dict["prediction_len"] = prediction_len
                return result_dict

            if max_prediction_len >= self.max_prediction_len:
                break
            max_prediction_len = min(max_prediction_len + 4, self.max_prediction_len)

        new_raw_ids = self.ctx_ids_cache[0, -prediction_len:]
        current_str = self.tokenizer.decode(new_raw_ids, skip_special_tokens=True)
        return {"new_str": current_str, "terminate": False,
                "prediction_len": prediction_len}

    def reset_resync_stats(self):
        """Reset per-sample resync counters."""
        self._step_count = 0
        self._resync_count = 0
        self._resync_lengths = []

    def get_resync_stats(self):
        """Return resync stats for this model."""
        return {
            "model": self.model_name,
            "num_watermarks": self.num_watermarks,
            "total_steps": self._step_count,
            "resync_steps": self._resync_count,
            "resync_lengths": list(self._resync_lengths),
        }

    def _get_next_tokens(self, logits):
        self._step_count += 1
        watermarked_prob = self._get_watermark_prob(logits)
        new_ids = torch.multinomial(watermarked_prob, num_samples=1)
        raw_new_id = new_ids.cpu()[0]
        raw_new_token = self.tokenizer.convert_ids_to_tokens(raw_new_id)[0]

        if not self.do_resync or (raw_new_id[0] in self.reserve_ids and not raw_new_token[-1].isdigit()):
            terminate = (raw_new_id[0].item() in self.stop_token_ids)
            result_dict = {
                "new_str": self.tokenizer.decode(raw_new_id, skip_special_tokens=True) if not terminate else "",
                "new_raw_tokens": [raw_new_token] if not terminate else [],
                "terminate": terminate,
            }

            generated_wm_name = self.watermark_index[int(self.watermark_choice_backup)].watermark_type
            generated_token = self.tokenizer.convert_ids_to_tokens(raw_new_id)[0] if not terminate else ""
            file_logger.info(f"GENERATE ({self.model_name}-{generated_wm_name}): '{generated_token}'")
            return result_dict

        # Resync: predict ahead then restore
        self._resync_count += 1
        self._save_state()
        predict_result = self._get_prediction_seq(new_ids)
        self._restore_state()
        self._resync_lengths.append(predict_result.get("prediction_len", 1))

        new_raw_ids = self.tokenizer(predict_result["new_str"], add_special_tokens=False)["input_ids"]
        new_raw_tokens = self.tokenizer.convert_ids_to_tokens(new_raw_ids)
        file_logger.info(f"GENERATE ({self.model_name}): {new_raw_tokens}")
        return {
            "new_str": predict_result["new_str"],
            "new_raw_tokens": new_raw_tokens,
            "terminate": predict_result["terminate"],
        }

    def _current_device(self):
        """Device for small state (ctx_ids_cache etc). Always GPU — only
        model weights move to CPU during offload."""
        return self.device

    def _setup_context(self, next_str, reset):
        dev = self._current_device()
        if reset:
            if self.is_instruct:
                # Apply chat template for instruct models
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": next_str[0]}],
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
                tokenization_output = self.tokenizer(
                    [formatted], padding=True, return_tensors="pt",
                    add_special_tokens=False)
            else:
                tokenization_output = self.tokenizer(next_str, padding=True, return_tensors="pt")
            input_ids = tokenization_output["input_ids"].to(dev)
            self.ctx_ids_cache = input_ids
            self.logits_generator.reset_state()
            for watermark in self.watermark_index.values():
                watermark.initialize_params(batch_size=1, keep_shifts=True)
        else:
            tokenization_output = self.tokenizer(
                next_str, padding=True, add_special_tokens=False, return_tensors="pt")
            input_ids = tokenization_output["input_ids"].to(dev)
            self.ctx_ids_cache = torch.cat([self.ctx_ids_cache, input_ids], dim=1)
            for watermark in self.watermark_index.values():
                watermark.update_params(step=input_ids.shape[1])

        file_logger.info(
            f"FILL ({self.model_name}): {self.tokenizer.convert_ids_to_tokens(input_ids[0])}")
        return input_ids

    def get_next_tokens(self, next_str: list[str], selected: bool = False, reset: bool = False):
        assert len(next_str) == 1, "Currently only supports batch_size=1"
        input_ids = self._setup_context(next_str=next_str, reset=reset)

        if reset:
            self._kv_len = 0

        if not selected:
            # Skip forward pass — ctx_ids_cache and watermark state are still updated
            # by _setup_context, only KV cache falls behind
            return None

        ctx_len = self.ctx_ids_cache.shape[1]
        if self._kv_len < ctx_len:
            # KV cache is behind — catch up with the missing tokens
            missing_ids = self.ctx_ids_cache[:, self._kv_len:]
            if self._kv_len == 0:
                self.logits_generator.reset_state()
            logits = self.logits_generator.get_logits(missing_ids)
        else:
            # KV cache is current — process only new tokens
            logits = self.logits_generator.get_logits(input_ids)

        self._kv_len = ctx_len
        logits = logits[:, :self.vocab_size]
        return self._get_next_tokens(logits)

    # ── CPU offload helpers ──────────────────────────────────

    def offload_to_cpu(self):
        """Move model weights to CPU. KV cache + small state stay on GPU."""
        self.logits_generator.offload_to_cpu()
        # ctx_ids_cache, reserve_ids, KV caches all stay on GPU — tiny

    def load_to_gpu(self, device=None):
        """Move model weights back to GPU. Small state is already there."""
        target = device or self.device
        self.logits_generator.load_to_gpu(target)
        self.device = target
