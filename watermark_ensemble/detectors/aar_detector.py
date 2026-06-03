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

from ..watermarks.aar import AARWatermark


class AARDetector(AARWatermark):
    """Detector for AAR (Aaronson/EXP) watermarks.

    Detection uses the exponential scoring: for each token t_i in the text,
    compute u_i = the seeded random value for that token at that position.
    Under the watermark, u values for selected tokens are biased high
    (since generation selects argmax u^(1/p)).

    Test statistic: S = sum(-log(1 - u_i)) for scored tokens.
    Under null: each -log(1-u) ~ Exp(1), so S ~ Gamma(n,1), E[S]=n, Var[S]=n.
    Z-score: (S - n) / sqrt(n).
    """

    def __init__(self, *args, z_threshold: float = 4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_threshold = z_threshold

    def _compute_scores(self, input_ids: torch.Tensor):
        """Compute per-token scores for the given token sequence.

        Returns a list of -log(1 - u[t_i]) values for each scored position.
        Positions before prefix_length are skipped (no context available).
        """
        scores = []
        seq_len = len(input_ids)

        for i in range(self.prefix_length, seq_len):
            # Build context window: tokens up to position i (exclusive)
            context = input_ids[:i].unsqueeze(0).to(self.device)
            # Get random u values for this context
            u_values = self._get_random_u(
                input_ids=context,
                prefix_length=self.prefix_length,
                vocab_size=self.vocab_size,
            )  # [1, vocab_size]

            # Score the actual token
            token_id = input_ids[i].item()
            u_token = u_values[0, token_id].item()
            # Clamp to avoid log(0)
            u_token = min(u_token, 1.0 - 1e-10)
            score = -torch.log(torch.tensor(1.0 - u_token)).item()
            scores.append(score)

        return scores

    def _score_sequence(self, input_ids: torch.Tensor):
        scores = self._compute_scores(input_ids)
        n = len(scores)
        if n == 0:
            raise ValueError("No tokens to score (sequence too short for prefix_length)")

        S = sum(scores)
        mean_score = S / n
        # Empirical variance — accounts for correlation between model token
        # preferences and hash-derived u values (the fixed-variance z-test
        # assumes Var=1 which underestimates true variance, causing false positives)
        var_score = sum((s - mean_score) ** 2 for s in scores) / (n - 1) if n > 1 else 1.0
        std_score = sqrt(var_score)
        # z-test: (mean - expected_mean) / (std / sqrt(n))
        z_score = (mean_score - 1.0) / (std_score / sqrt(n)) if std_score > 0 else 0.0
        # One-sided p-value using normal distribution (matches original AAR paper)
        p_value = scipy.stats.norm.sf(z_score)

        return {
            "num_tokens_scored": n,
            "sum_score": S,
            "mean_score": mean_score,
            "std_score": std_score,
            "z_score": z_score,
            "p_value": p_value,
        }

    def detect(self, tokenized_text, z_threshold: float = None) -> dict:
        """Score tokenized text and return detection results.

        Args:
            tokenized_text: 1D tensor of token IDs.
            z_threshold: Override detection threshold (default: self.z_threshold).

        Returns:
            Dict with num_tokens_scored, sum_score, mean_score, z_score, p_value, prediction.
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
