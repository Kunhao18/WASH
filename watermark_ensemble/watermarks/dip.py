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
import random
import hashlib

from torch.nn import functional as F

from .base import BaseWatermark


class DIPWatermark(BaseWatermark):
    """Disentanglement-based watermark using vocabulary shuffling and logit reweighting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        random.seed(self.config["key"])
        self.hash_key = random.getrandbits(1024).to_bytes(128, "big")
        self.alpha = self.config["alpha"]
        self.prefix_length = self.config["prefix_length"]

    def _get_rng_seed(self, context_code: bytes) -> int:
        m = hashlib.sha256()
        m.update(context_code)
        m.update(self.hash_key)
        full_hash = m.digest()
        return int.from_bytes(full_hash, "big") % (2**32 - 1)

    def _extract_context_code(self, context, prefix_length):
        if prefix_length == 0:
            return context
        return context[:, -prefix_length:]

    def _get_seeds_for_cipher(self, input_ids, prefix_length):
        context_codes = self._extract_context_code(input_ids, prefix_length)
        seeds = []
        for i in range(context_codes.size(0)):
            context_code = context_codes[i].detach().cpu().numpy().tobytes()
            seeds.append(self._get_rng_seed(context_code))
        return seeds

    def _from_random(self, rng, vocab_size):
        return torch.stack([
            torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
            for i in range(len(rng))
        ])

    def _reweight_logits(self, shuffle, p_logits, alpha):
        unshuffle = torch.argsort(shuffle, dim=-1)

        s_p_logits = torch.gather(p_logits, -1, shuffle)
        s_log_cumsum = torch.logcumsumexp(s_p_logits, dim=-1)
        s_log_cumsum = s_log_cumsum - s_log_cumsum[..., -1:]
        s_cumsum = torch.exp(s_log_cumsum)
        s_p = F.softmax(s_p_logits, dim=-1)

        # Boundary 1
        boundary_1 = torch.argmax((s_cumsum > alpha).to(torch.int), dim=-1, keepdim=True)
        p_boundary_1 = torch.gather(s_p, -1, boundary_1)
        portion_in_right_1 = (torch.gather(s_cumsum, -1, boundary_1) - alpha) / p_boundary_1
        portion_in_right_1 = torch.clamp(portion_in_right_1, 0, 1)
        s_all_portion_in_right_1 = (s_cumsum > alpha).type_as(p_logits)
        s_all_portion_in_right_1.scatter_(-1, boundary_1, portion_in_right_1)

        # Boundary 2
        boundary_2 = torch.argmax((s_cumsum > (1 - alpha)).to(torch.int), dim=-1, keepdim=True)
        p_boundary_2 = torch.gather(s_p, -1, boundary_2)
        portion_in_right_2 = (torch.gather(s_cumsum, -1, boundary_2) - (1 - alpha)) / p_boundary_2
        portion_in_right_2 = torch.clamp(portion_in_right_2, 0, 1)
        s_all_portion_in_right_2 = (s_cumsum > (1 - alpha)).type_as(p_logits)
        s_all_portion_in_right_2.scatter_(-1, boundary_2, portion_in_right_2)

        s_all_portion_in_right = s_all_portion_in_right_2 / 2 + s_all_portion_in_right_1 / 2
        s_shift_logits = torch.log(s_all_portion_in_right)
        shift_logits = torch.gather(s_shift_logits, -1, unshuffle)

        return p_logits + shift_logits

    def _apply_watermark(self, input_ids, scores, alpha, prefix_length):
        seeds = self._get_seeds_for_cipher(input_ids, prefix_length)
        rng = [torch.Generator(device=scores.device).manual_seed(seed) for seed in seeds]
        shuffle = self._from_random(rng, scores.size(1))
        return self._reweight_logits(shuffle, scores, alpha)

    def initialize_params(self, **kwargs):
        pass

    def update_params(self, **kwargs):
        pass

    def apply_watermark(self, logits, input_ids, **kwargs):
        if input_ids.shape[-1] < self.prefix_length:
            raise ValueError("Input sequence shorter than prefix_length")

        reweighted_scores = self._apply_watermark(
            input_ids, logits, self.alpha, self.prefix_length)
        probs = self._sampling(reweighted_scores)
        return probs
