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

import copy
import torch

from torch.nn import functional as F

from .mersenne import MersenneRNG
from .base import BaseWatermark


class KTHWatermark(BaseWatermark):
    """Kuditipudi et al. token hiding with reweighting watermark."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = self.config["key"]
        self.keylen = self.config["keylen"]
        self.prefix_length = self.config["prefix_length"]

        rng = MersenneRNG(self.key)
        self.xi = torch.tensor(
            [rng.rand() for _ in range(self.keylen * self.vocab_size)]
        ).view(self.keylen, self.vocab_size).to(self.device)

        self.shifts = None
        self.cnts = None

    def initialize_params(self, batch_size, keep_shifts=False, **kwargs):
        if not (keep_shifts and self.shifts is not None):
            self.shifts = torch.randint(self.keylen, (batch_size,), device=self.device)
        self.cnts = torch.zeros(batch_size, dtype=torch.long, device=self.device)

    def update_params(self, step: int = 1, **kwargs):
        self.cnts += step

    def save_state(self, **kwargs):
        self._cnts_backup = copy.deepcopy(self.cnts)

    def restore_state(self, **kwargs):
        self.cnts = self._cnts_backup

    def apply_watermark(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        active_indices: torch.Tensor = None,
        normalize: bool = True,
        **kwargs
    ) -> torch.Tensor:
        if input_ids.shape[-1] < self.prefix_length:
            return logits

        probs = self._sampling(logits)

        if active_indices is not None:
            xi_indices = (self.shifts[active_indices] + self.cnts[active_indices]) % self.keylen
        else:
            xi_indices = (self.shifts + self.cnts) % self.keylen
        xi_batch = self.xi[xi_indices]

        eps = torch.finfo(probs.dtype).eps
        probs = torch.where(probs <= 0, torch.tensor(eps, device=probs.device), probs)
        final_probs = xi_batch ** (1 / probs)
        if normalize:
            final_probs = F.normalize(final_probs, p=1, dim=-1)

        return final_probs
