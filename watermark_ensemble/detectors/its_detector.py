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
#
# ITS (Inverse Transform Sampling) detector. This follows the original
# MarkLLM "EXPEdit" detector: an edit-distance (Levenshtein) alignment test
# statistic with a permutation test over random keys. Detection returns a
# p-value only (the original reports no z-score), and the full permutation /
# shift search is used (no known/"oracle" shift shortcut).

import os
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from ..watermarks.its import ITSWatermark
from ._dp_worker import dp_levenshtein_cost, _its_detect_null_worker


def _num_workers():
    cpus = os.cpu_count() or 1
    return min(cpus, 8)


class ITSDetector(ITSWatermark):
    """Detector for ITS (Inverse Transform Sampling) watermarks.

    Reproduces the MarkLLM EXPEdit permutation test: the test statistic is the
    minimum Levenshtein alignment cost between the token-rank sequence and the
    cyclic shifts of the pseudo-random key; the null distribution is built from
    ``n_runs`` random keys. The watermark is declared present when the resulting
    p-value falls below ``p_threshold``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pi_inv = torch.zeros_like(self.pi)
        self.pi_inv[self.pi] = torch.arange(self.vocab_size, device=self.device)

    def _get_token_ranks_np(self, tokenized_text):
        """Normalized token ranks (rank / vocab_size) as a numpy array."""
        t = torch.as_tensor(tokenized_text, dtype=torch.long)
        token_ranks = self.pi_inv.cpu()[t].float() / self.vocab_size
        return token_ranks.numpy().copy()

    def calculate(self, tokenized_text, gamma: float = 0.0, max_shifts: int = 0):
        """Minimum over cyclic shifts of the Levenshtein alignment cost.

        ``max_shifts=0`` searches every shift (the full, original test);
        a positive value samples that many random shifts for speed.
        """
        xi_np = self.xi.cpu().numpy()
        keylen = xi_np.shape[0]

        token_ranks_np = self._get_token_ranks_np(tokenized_text)
        n = len(token_ranks_np)
        pos = np.arange(n)

        if max_shifts > 0 and max_shifts < keylen:
            shifts = np.random.choice(keylen, size=max_shifts, replace=False)
        else:
            shifts = np.arange(keylen)

        idx = (shifts[:, None] + pos[None, :]) % keylen
        xi_vals = xi_np[idx].squeeze(-1)
        distances = np.abs(token_ranks_np[None, :] - xi_vals)
        distances = np.clip(distances, 0, 1.0 - 1e-7)
        cost_matrix = np.log1p(-distances).astype(np.float32)

        scores = dp_levenshtein_cost(cost_matrix, gamma)
        return float(scores.min())

    def detect(self, tokenized_text, n_runs=100, max_shifts=0, n_workers=0,
               gamma=0.0, p_threshold=0.05):
        """Permutation-test detection. Returns a dict with the p-value only.

        Returns:
            {"num_tokens_scored", "p_value", "prediction"}.
        """
        token_ranks_np = self._get_token_ranks_np(tokenized_text)
        n = len(token_ranks_np)

        test_result = self.calculate(tokenized_text, max_shifts=max_shifts,
                                     gamma=gamma)

        workers = n_workers if n_workers > 0 else _num_workers()
        workers = min(workers, n_runs)
        runs_per_worker = [n_runs // workers] * workers
        for i in range(n_runs % workers):
            runs_per_worker[i] += 1

        base_seed = int.from_bytes(os.urandom(4), "big")
        args_list = [
            (token_ranks_np, self.keylen, max_shifts, n,
             runs_per_worker[w], test_result, base_seed + w, gamma)
            for w in range(workers)
        ]

        p_val = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for count in pool.map(_its_detect_null_worker, args_list):
                p_val += count

        p_value = (p_val + 1.0) / (n_runs + 1.0)
        return {
            "num_tokens_scored": float(n),
            "p_value": float(p_value),
            "prediction": bool(p_value < p_threshold),
        }
