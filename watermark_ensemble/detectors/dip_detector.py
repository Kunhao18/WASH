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
import scipy.stats

from math import sqrt

from ..watermarks.dip import DIPWatermark


class DIPDetector(DIPWatermark):
    """Detector for DIP (DiPmark) watermarks.

    Detection reconstructs the vocabulary shuffle at each position and checks
    whether the generated token falls in the favored region (first alpha fraction
    of the shuffled vocabulary). Under the watermark, tokens are biased toward
    this region. A z-test on the fraction of tokens in the favored region
    determines watermark presence.

    Under null hypothesis: P(token in favored region) = alpha.
    Under watermark: P(token in favored region) > alpha.
    """

    def __init__(self, *args, z_threshold: float = 4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_threshold = z_threshold

    def _check_token_in_greenlist(self, input_ids_context: torch.Tensor, token_id: int) -> bool:
        """Check if a token falls in the favored (green) region of the DIP shuffle.

        Args:
            input_ids_context: Context token IDs used to derive the shuffle seed.
            token_id: The token to check.

        Returns:
            True if the token is in the favored region.
        """
        seeds = self._get_seeds_for_cipher(input_ids_context, self.prefix_length)
        rng = [torch.Generator(device=self.device).manual_seed(seed) for seed in seeds]
        shuffle = self._from_random(rng, self.vocab_size)  # [1, vocab_size]

        # DIP's _reweight_logits suppresses the first alpha fraction of the shuffled
        # vocab (weight→0) and favors the last alpha fraction (weight→1). So the
        # "green" (favored) region is the LAST alpha fraction of the shuffle.
        greenlist_size = int(self.vocab_size * self.alpha)

        # Find position of token_id in the shuffle
        token_pos = (shuffle[0] == token_id).nonzero(as_tuple=True)[0]
        if len(token_pos) == 0:
            return False
        pos = token_pos[0].item()
        return pos >= self.vocab_size - greenlist_size

    def _score_sequence(self, input_ids: torch.Tensor):
        """Score a token sequence for DIP watermark presence."""
        seq_len = len(input_ids)
        green_count = 0
        total_scored = 0

        for i in range(self.prefix_length, seq_len):
            context = input_ids[:i].unsqueeze(0).to(self.device)
            token_id = input_ids[i].item()
            if self._check_token_in_greenlist(context, token_id):
                green_count += 1
            total_scored += 1

        if total_scored == 0:
            raise ValueError("No tokens to score (sequence too short for prefix_length)")

        green_fraction = green_count / total_scored
        # Under null: expected fraction = alpha
        # z-test: (observed - expected) / sqrt(expected * (1-expected) / n)
        expected = self.alpha
        z_score = (green_fraction - expected) / sqrt(expected * (1 - expected) / total_scored)
        p_value = scipy.stats.norm.sf(z_score)

        return {
            "num_tokens_scored": total_scored,
            "num_green_tokens": green_count,
            "green_fraction": green_fraction,
            "z_score": z_score,
            "p_value": p_value,
        }

    def detect(self, tokenized_text, z_threshold: float = None) -> dict:
        """Score tokenized text and return detection results.

        Args:
            tokenized_text: 1D tensor of token IDs.
            z_threshold: Override detection threshold (default: self.z_threshold).

        Returns:
            Dict with num_tokens_scored, num_green_tokens, green_fraction,
            z_score, p_value, prediction.
        """
        if isinstance(tokenized_text, list):
            tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)
        output_dict = self._score_sequence(tokenized_text)
        z_threshold = z_threshold if z_threshold else self.z_threshold
        output_dict["prediction"] = output_dict["z_score"] > z_threshold
        for key, value in output_dict.items():
            if isinstance(value, int):
                output_dict[key] = float(value)
        return output_dict
