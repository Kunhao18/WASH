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

from .mersenne import MersenneRNG
from .base import BaseWatermark


class WaterbagWatermark(BaseWatermark):
    """Enhanced KGW watermark with multiple key list support and indicator-based split."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hash_key = self.config["hash_key"]
        self.prefix_length = self.config["prefix_length"]
        self.gamma = self.config["gamma"]
        self.delta = self.config["delta"]
        self.scheme = self.config["scheme"]
        self.keylen = self.config["keylen"]
        self.f_scheme_map = {
            "time": self._f_time,
            "additive": self._f_additive,
            "skip": self._f_skip,
            "min": self._f_min,
        }

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.hash_key)

        mersenne_rng = MersenneRNG(seed=42)
        if self.keylen != 0:
            self.key_list = torch.tensor(
                [int(mersenne_rng.randint() * 10e8) for _ in range(self.keylen)]
            ).to(self.device)
        else:
            self.key_list = torch.tensor(
                [int(mersenne_rng.randint() * 10e8)]
            ).to(self.device)

        assert all(isinstance(key.item(), int) for key in self.key_list)
        self.prfs = torch.stack(
            [torch.randperm(self.vocab_size, device=self.device,
                            generator=self.rng.manual_seed(key.item()))
             for key in self.key_list]
        ).to(self.device)

        self.indices = None
        self.indicator_list = None

    def _f(self, input_ids, prefix_length, prf, vocab_size):
        return self.f_scheme_map[self.scheme](input_ids, prefix_length, prf, vocab_size)

    def _f_additive(self, input_ids, prefix_length, prf, vocab_size):
        batch_size, sequence_length = input_ids.shape
        assert sequence_length >= prefix_length
        additive_result = torch.ones(batch_size, device=input_ids.device)
        for i in range(prefix_length):
            additive_result += input_ids[:, -1 - i].float()
        indices = (additive_result.long() % vocab_size).unsqueeze(1)
        return torch.gather(prf, 1, indices).squeeze(1)

    def _f_skip(self, input_ids, prefix_length, prf, vocab_size):
        batch_size, seq_length = input_ids.shape
        assert seq_length >= prefix_length
        skip_tokens = input_ids[:, -prefix_length]
        return prf[skip_tokens]

    def _f_min(self, input_ids, prefix_length, prf, vocab_size):
        batch_size, seq_length = input_ids.shape
        assert seq_length >= prefix_length
        last_tokens = input_ids[:, -prefix_length:]
        prf_values = prf[last_tokens]
        return prf_values.min(dim=1).values

    def _f_time(self, input_ids, prefix_length, prf, vocab_size):
        batch_size, sequence_length = input_ids.shape
        time_result = torch.ones(batch_size, device=input_ids.device)
        for i in range(prefix_length):
            time_result *= input_ids[:, -1 - i].float()
        indices = (time_result.long() % vocab_size).unsqueeze(1)
        return torch.gather(prf, 1, indices).squeeze(1)

    def _get_greenlist_ids(self, input_ids, gamma, prf, vocab_size, prefix_length, keys, indicators):
        time_results = self._f(input_ids, prefix_length, prf, vocab_size)
        seeds = ((keys * time_results) % vocab_size).to(self.device)

        greenlist_size = int(vocab_size * gamma)
        rng_cuda = torch.Generator(device=self.device)

        vocab_permutations = torch.stack(
            [
                torch.randperm(
                    vocab_size, device=self.device,
                    generator=rng_cuda.manual_seed(seed.item())
                )
                for seed in seeds
            ],
            dim=0,
        )

        effective_vocab_size = vocab_size - (vocab_size % 2)
        greenlist_ids = torch.where(
            indicators.unsqueeze(1) == 1,
            vocab_permutations[:, :greenlist_size],
            vocab_permutations[:, greenlist_size:effective_vocab_size]
        )
        return greenlist_ids

    def _calc_greenlist_mask(self, scores, greenlist_token_ids):
        batch_size, vocab_size = scores.shape
        green_tokens_mask = torch.zeros(batch_size, vocab_size, device=scores.device, dtype=torch.bool)
        green_tokens_mask.scatter_(1, greenlist_token_ids, True)
        return green_tokens_mask

    def _bias_greenlist_logits(self, scores, greenlist_mask, greenlist_bias):
        _scores = scores.clone()
        _scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return _scores

    def initialize_params(self, batch_size, **kwargs):
        self.indices = torch.randint(0, self.keylen, (batch_size,)).to(self.device)
        self.indicator_list = torch.randint(0, 2, (batch_size,)).to(self.device)

    def update_params(self, **kwargs):
        pass

    def apply_watermark(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        active_indices: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        if input_ids.shape[-1] < self.prefix_length:
            return logits

        if active_indices is not None:
            indices = self.indices[active_indices]
            selected_indicators = self.indicator_list[active_indices]
        else:
            indices = self.indices
            selected_indicators = self.indicator_list
        selected_keys = self.key_list[indices]
        selected_prfs = self.prfs[indices]

        batched_greenlist_ids = self._get_greenlist_ids(
            input_ids, self.gamma, selected_prfs, self.vocab_size,
            self.prefix_length, selected_keys, selected_indicators)
        green_tokens_mask = self._calc_greenlist_mask(logits, batched_greenlist_ids)
        logits = self._bias_greenlist_logits(logits, green_tokens_mask, self.delta)
        probs = self._sampling(logits)
        return probs
