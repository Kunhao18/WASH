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

from torch.nn import functional as F

from .base import BaseWatermark


class AARWatermark(BaseWatermark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hash_key = self.config["hash_key"]
        self.prefix_length = self.config["prefix_length"]
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.hash_key)

    def _get_random_u(
        self,
        input_ids: torch.Tensor,
        prefix_length: int,
        vocab_size: int,
    ) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        time_result = torch.ones(batch_size, device=input_ids.device)
        for i in range(prefix_length):
            time_result *= input_ids[:, -1 - i].long()
        prev_tokens = time_result % vocab_size

        random_u_batch = torch.stack(
            [
                torch.rand(
                    vocab_size,
                    device=self.device,
                    generator=self.rng.manual_seed(int(self.hash_key * seed.item())),
                )
                for seed in prev_tokens
            ],
            dim=0,
        )
        return random_u_batch

    def initialize_params(self, **kwargs):
        pass

    def update_params(self, **kwargs):
        pass

    def apply_watermark(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        normalize: bool = True,
        **kwargs
    ) -> torch.Tensor:
        if input_ids.shape[-1] < self.prefix_length:
            raise ValueError("Input sequence shorter than prefix_length")

        batched_random_u = self._get_random_u(
            input_ids=input_ids,
            prefix_length=self.prefix_length,
            vocab_size=self.vocab_size,
        )
        probs = self._sampling(logits)

        eps = torch.finfo(logits.dtype).eps
        probs = torch.where(probs <= 0, torch.tensor(eps, device=probs.device), probs)
        final_probs = batched_random_u ** (1 / probs)
        if normalize:
            final_probs = F.normalize(final_probs, p=1, dim=-1)

        return final_probs
