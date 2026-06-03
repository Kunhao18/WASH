# Copyright 2024 THU BPM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from abc import ABC
from torch.nn import functional as F


class BaseWatermark(ABC):
    def __init__(
        self,
        temperature: float,
        top_k: int,
        vocab_size: int,
        watermark_type: str,
        config: dict,
        device: str,
    ):
        self.device = torch.device(device)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = 1.0
        self.vocab_size = vocab_size
        self.watermark_type = watermark_type
        self.config = config

    @torch.no_grad()
    def _sampling(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature, top-k and top-p sampling to get probability distribution."""
        assert self.temperature > 0, "Temperature must be positive"

        _logits = logits / self.temperature

        if self.top_k is not None and self.top_k > 0:
            top_k = min(self.top_k, _logits.size(-1))
            indices_to_remove = _logits < torch.topk(_logits, top_k)[0][..., -1, None]
            _logits[indices_to_remove] = float("-inf")

        if self.top_p is not None and 0 < self.top_p < 1:
            sorted_logits, sorted_indices = torch.sort(_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            if sorted_indices_to_remove[..., 1:].any():
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            _logits[indices_to_remove] = float("-inf")

        probs = F.softmax(_logits, dim=-1)
        return probs

    def initialize_params(self, **kwargs):
        raise NotImplementedError

    def update_params(self, **kwargs):
        raise NotImplementedError

    def save_state(self, **kwargs):
        pass

    def restore_state(self, **kwargs):
        pass

    def apply_watermark(self, **kwargs):
        raise NotImplementedError
