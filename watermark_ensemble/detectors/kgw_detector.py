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
import collections
import scipy.stats

from math import sqrt
from functools import lru_cache
from nltk.util import ngrams

from ..watermarks.kgw import KGWWatermark


class KGWDetector(KGWWatermark):
    """Detector for KGW watermarks. Replicates the greenlist generation
    logic to detect watermark presence via z-score hypothesis testing."""

    def __init__(
        self,
        *args,
        z_threshold: float = 4.0,
        ignore_repeated_ngrams: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.z_threshold = z_threshold
        self.ignore_repeated_ngrams = ignore_repeated_ngrams

    def _compute_z_score(self, observed_count, T):
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        return numer / denom

    def _compute_p_value(self, observed_count, T):
        return scipy.stats.binom.sf(observed_count, T, self.gamma)

    @lru_cache(maxsize=2**32)
    def _get_ngram_score_cached(self, prefix: tuple, target: int):
        greenlist_ids = self._get_greenlist_ids(
            input_ids=torch.tensor(prefix, dtype=torch.long).unsqueeze(0).to(self.device),
            gamma=self.gamma,
            prf=self.prf,
            vocab_size=self.vocab_size,
            prefix_length=len(prefix)
        )
        return target in greenlist_ids

    def _score_ngrams_in_passage(self, input_ids: torch.Tensor):
        if len(input_ids) - self.prefix_length < 1:
            raise ValueError(
                f"Must have at least 1 token to score after "
                f"the first {self.prefix_length} prefix tokens."
            )

        token_ngram_generator = ngrams(input_ids.cpu().tolist(), self.prefix_length + 1)
        frequencies_table = collections.Counter(token_ngram_generator)
        ngram_to_watermark_lookup = {}
        for ngram_example in frequencies_table.keys():
            prefix = ngram_example[:-1]
            target = ngram_example[-1]
            ngram_to_watermark_lookup[ngram_example] = self._get_ngram_score_cached(prefix, target)

        return ngram_to_watermark_lookup, frequencies_table

    def _get_green_at_T_booleans(self, input_ids, ngram_to_watermark_lookup):
        green_token_mask, green_token_mask_unique, offsets = [], [], []
        used_ngrams = {}
        unique_ngram_idx = 0
        ngram_examples = ngrams(input_ids.cpu().tolist(), self.prefix_length + 1)

        for idx, ngram_example in enumerate(ngram_examples):
            green_token_mask.append(ngram_to_watermark_lookup[ngram_example])
            if self.ignore_repeated_ngrams:
                if ngram_example not in used_ngrams:
                    used_ngrams[ngram_example] = True
                    unique_ngram_idx += 1
                    green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
            else:
                green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
                unique_ngram_idx += 1
            offsets.append(unique_ngram_idx - 1)
        return (
            torch.tensor(green_token_mask),
            torch.tensor(green_token_mask_unique),
            torch.tensor(offsets),
        )

    def _score_sequence(self, input_ids: torch.Tensor):
        ngram_to_watermark_lookup, frequencies_table = self._score_ngrams_in_passage(input_ids)
        green_token_mask, green_unique, offsets = self._get_green_at_T_booleans(
            input_ids, ngram_to_watermark_lookup)

        if self.ignore_repeated_ngrams:
            num_tokens_scored = len(frequencies_table.keys())
            green_token_count = sum(ngram_to_watermark_lookup.values())
        else:
            num_tokens_scored = sum(frequencies_table.values())
            green_token_count = sum(
                freq * outcome
                for freq, outcome in zip(
                    frequencies_table.values(), ngram_to_watermark_lookup.values()
                )
            )

        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        p_value = self._compute_p_value(green_token_count, num_tokens_scored)
        return {
            "num_tokens_scored": num_tokens_scored,
            "num_green_tokens": green_token_count,
            "green_fraction": green_token_count / num_tokens_scored,
            "z_score": z_score,
            "p_value": p_value,
        }

    def detect(self, tokenized_text, z_threshold: float = None) -> dict:
        """Score tokenized text and return detection results."""
        output_dict = self._score_sequence(tokenized_text)
        z_threshold = z_threshold if z_threshold else self.z_threshold
        output_dict["prediction"] = output_dict["z_score"] > z_threshold
        for key, value in output_dict.items():
            if isinstance(value, int):
                output_dict[key] = float(value)
        return output_dict
