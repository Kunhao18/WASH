import json
import time
import torch

from concurrent.futures import ThreadPoolExecutor, as_completed

from .model_manager import ModelManager
from ..utils.logging import file_logger


# WASH uses the standard ensemble manager; ``max_prediction_len`` controls the
# block length (WASH uses 32). Alternative attack managers (demark, toblend)
# are not part of this release.
METHOD_MAP = {
    "ensemble": ModelManager,
}


class GenerationManager:
    """Coordinates multiple model managers for ensemble (WASH) generation.

    Two execution modes:

      * **parallel** (``cpu_offload=False``, default) — each model stays resident
        on its own GPU (set per model via ``device`` in the config) and the
        ensemble members run concurrently. Fastest; needs one GPU per model.
      * **sequential** (``cpu_offload=True``) — all models share a single GPU.
        At each step the chosen model's weights are loaded to the GPU and the
        previous one is offloaded to CPU (dynamic offloading). Runs in a
        resource-constrained, single-GPU setting; slower.
    """

    def __init__(
        self,
        mix_config_path: str,
        temperature: float = 1.0,
        top_k: int = 10,
        max_prediction_len: int = 32,
        method: str = "ensemble",
        device: str = "cuda",
        cpu_offload: bool = False,
        dtype=None,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.max_prediction_len = max_prediction_len
        self.method = method
        self.device = device
        self.cpu_offload = cpu_offload
        self.dtype = dtype

        self.mix_config = None
        self.vocab_config = None
        self.model_managers = {}
        self.manager_index = {}
        self.manager_weights = None
        self._executor = None
        self._gpu_resident = None  # index of model currently on GPU (offload mode)
        self._load_config(mix_config_path)

    def _load_config(self, config_path: str):
        with open(config_path) as f:
            self.mix_config = json.load(f)

        # Vocab
        vocab_path = self.mix_config["vocab_config_path"]
        with open(vocab_path, encoding="utf8") as f_in:
            self.vocab_config = json.load(f_in)
        self.common_vocab = self.vocab_config["common_vocab"]

        # Model managers
        manager_weights = []
        self.manager_index = {}
        ManagerClass = METHOD_MAP[self.method]

        for idx, (model_name, model_config) in enumerate(self.mix_config["models"].items()):
            # In cpu_offload mode, all models share cuda:0
            if self.cpu_offload:
                model_device = "cuda:0"
            else:
                model_device = model_config.get("device", self.device)
            extra_kwargs = {}
            if model_config.get("is_instruct", False):
                extra_kwargs["is_instruct"] = True
            if self.dtype is not None:
                extra_kwargs["dtype"] = self.dtype
            current_manager = ManagerClass(
                model_name=model_name,
                model_path=model_config["model_path"],
                max_prediction_len=self.max_prediction_len,
                device=model_device,
                **extra_kwargs,
            )
            current_manager.setup_watermarks(
                watermarks=model_config["watermarks"],
                temperature=self.temperature,
                top_k=self.top_k,
            )
            current_manager.setup_vocab(common_vocab=self.common_vocab)
            self.model_managers[model_name] = current_manager
            self.manager_index[idx] = current_manager
            manager_weights.append(current_manager.num_watermarks)

            # In cpu_offload mode, load one model at a time on GPU then offload
            if self.cpu_offload:
                current_manager.offload_to_cpu()

        self.manager_weights = torch.tensor(manager_weights, dtype=torch.float)

    def _ensure_executor(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=len(self.model_managers))

    def _get_next_token(self, next_str, reset=False):
        if self.cpu_offload:
            return self._get_next_token_offload(next_str, reset)
        return self._get_next_token_parallel(next_str, reset)

    def _get_next_token_parallel(self, next_str, reset=False):
        model_choice = torch.multinomial(self.manager_weights, num_samples=1, replacement=True)[0]

        self._ensure_executor()
        future_to_meta = {}
        for manager_idx, model_manager in self.manager_index.items():
            model_name = model_manager.model_name
            current_selected = (int(model_choice) == manager_idx)
            fut = self._executor.submit(
                model_manager.get_next_tokens,
                [next_str], current_selected, reset
            )
            future_to_meta[fut] = (manager_idx, model_name, current_selected)

        for fut in as_completed(future_to_meta):
            _ = fut.result()

        result_dict = {}
        for fut in future_to_meta:
            manager_idx, model_name, current_selected = future_to_meta[fut]
            if current_selected:
                result_dict = fut.result()
                result_dict["model_name"] = model_name

        return result_dict

    def _get_next_token_offload(self, next_str, reset=False):
        """CPU-offload variant: one model on GPU at a time, hot-cache swap."""
        model_choice = int(torch.multinomial(
            self.manager_weights, num_samples=1, replacement=True)[0])

        # Swap if needed: offload current GPU model, load selected model
        if self._gpu_resident is not None and self._gpu_resident != model_choice:
            self.manager_index[self._gpu_resident].offload_to_cpu()
        if self._gpu_resident != model_choice:
            self.manager_index[model_choice].load_to_gpu("cuda:0")
            self._gpu_resident = model_choice

        # Run all managers sequentially
        result_dict = {}
        for manager_idx, model_manager in self.manager_index.items():
            current_selected = (manager_idx == model_choice)
            res = model_manager.get_next_tokens(
                [next_str], current_selected, reset)
            if current_selected:
                result_dict = res
                result_dict["model_name"] = model_manager.model_name

        return result_dict

    def run(self, context: str, max_len: int, stop_sequences: list[str] = None):
        """Generate text from the given context using the watermark ensemble.

        Args:
            context: The prompt/context string.
            max_len: Maximum number of tokens to generate.
            stop_sequences: Optional list of strings that trigger early stopping.

        Returns:
            Dict with generation_result, generation_length, generation_time, cuda_generation_time.
        """
        active = True
        next_str = context
        generation = ""

        # Reset per-sample resync stats
        for mm in self.model_managers.values():
            if hasattr(mm, "reset_resync_stats"):
                mm.reset_resync_stats()

        count = 0
        reset = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        start_time = time.time()

        while active and count < max_len:
            file_logger.info(f"---GENERATION {count}---")
            file_logger.info(f"*GLOBAL* GENERATION: {generation}")

            result_dict = self._get_next_token(next_str, reset)
            next_str = result_dict["new_str"]
            file_logger.info(f"*GLOBAL* NEXT STR: '{next_str}' ({result_dict['model_name']})")
            generation += next_str
            active = not result_dict["terminate"]

            # Check stop sequences
            if stop_sequences:
                for seq in stop_sequences:
                    if seq in generation:
                        active = False
                        break

            reset = False
            count += max(1, len(result_dict.get("new_raw_tokens", [next_str])))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        end_time = time.time()

        result = {
            "generation_result": generation,
            "generation_length": count,
            "generation_time": end_time - start_time,
            "cuda_generation_time": t1 - t0,
        }

        # Collect resync stats
        total_steps = 0
        total_resyncs = 0
        all_resync_lengths = []
        specialist_counts = []  # num_watermarks for each resync event
        for mm in self.model_managers.values():
            if not hasattr(mm, "get_resync_stats"):
                continue
            s = mm.get_resync_stats()
            total_steps += s["total_steps"]
            total_resyncs += s["resync_steps"]
            all_resync_lengths.extend(s["resync_lengths"])
            specialist_counts.extend(
                [s["num_watermarks"]] * s["resync_steps"])
        if total_steps > 0:
            result["resync_stats"] = {
                "total_steps": total_steps,
                "resync_steps": total_resyncs,
                "resync_fraction": total_resyncs / total_steps if total_steps else 0,
                "avg_resync_length": (sum(all_resync_lengths) / len(all_resync_lengths)
                                      if all_resync_lengths else 0),
                "avg_specialist_count": (sum(specialist_counts) / len(specialist_counts)
                                         if specialist_counts else 0),
            }

        return result
