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

from .base import BaseWatermark


class ITSWatermark(BaseWatermark):
    """Inverse Transform Sampling watermark using permutation-based sampling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = self.config["key"]
        self.keylen = self.config["keylen"]
        self.prefix_length = self.config["prefix_length"]

        rng = torch.Generator(device=self.device).manual_seed(self.key)
        self.xi = torch.rand((self.keylen, 1), generator=rng, device=self.device)
        self.pi = torch.randperm(self.vocab_size, generator=rng, device=self.device)

        self.shifts = None
        self.cnts = None

    def _transform_sampling(self, probs, pi, xi):
        cdf = torch.cumsum(torch.gather(probs, 1, pi), 1)
        indices = torch.searchsorted(cdf, xi)
        # Clamp to avoid OOB when xi >= cdf[-1] due to float precision
        indices = indices.clamp(max=cdf.shape[1] - 1)
        return torch.gather(pi, 1, indices)

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
        **kwargs
    ) -> torch.Tensor:
        if input_ids.shape[-1] < self.prefix_length:
            raise ValueError("Input sequence shorter than prefix_length")

        batch_size = input_ids.shape[0]
        probs = self._sampling(logits)

        if active_indices is not None:
            xi_indices = (self.shifts[active_indices] + self.cnts[active_indices]) % self.keylen
        else:
            xi_indices = (self.shifts + self.cnts) % self.keylen
        xi_batch = self.xi[xi_indices]
        pi_batch = self.pi.repeat(batch_size, 1)

        sampled_indices = self._transform_sampling(probs, pi_batch, xi_batch).squeeze(1)
        final_probs = torch.zeros(probs.shape, device=self.device)
        final_probs.scatter_(1, sampled_indices.unsqueeze(1), 1.0)
        return final_probs
