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
# KTH detector. This follows the original MarkLLM "EXP" detector: a
# Levenshtein alignment test statistic with a permutation test over random
# keys. Detection returns a p-value only, and the full permutation / shift
# search is used (no known/"oracle" shift shortcut).

import os
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from ..watermarks.kth import KTHWatermark
from ._dp_worker import dp_levenshtein_lazy, _kth_detect_null_worker


def _num_workers():
    cpus = os.cpu_count() or 1
    return min(cpus, 8)


class KTHDetector(KTHWatermark):
    """Detector for KTH watermarks using Levenshtein-based alignment scoring.

    Reproduces the MarkLLM EXP permutation test: the test statistic is the
    minimum Levenshtein alignment cost over the cyclic shifts of the real key;
    the null distribution is built from ``n_runs`` random keys. The watermark is
    declared present when the resulting p-value falls below ``p_threshold``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _to_numpy(self, tokenized_text):
        """Convert tokenized text and the key to numpy for worker processes."""
        t = torch.as_tensor(tokenized_text, device="cpu", dtype=torch.long)
        tokens_np = t.numpy().copy()
        xi_np = self.xi.cpu().numpy().copy()
        return tokens_np, xi_np

    def calculate(self, tokenized_text, gamma: float = 0.0, max_shifts: int = 0):
        """Minimum over cyclic shifts of the Levenshtein alignment cost.

        ``max_shifts=0`` searches every shift (the full, original test);
        a positive value samples that many random shifts for speed.
        """
        xi_np = self.xi.cpu().numpy()
        keylen = xi_np.shape[0]

        t = torch.as_tensor(tokenized_text, dtype=torch.long)
        tokens_np = t.numpy().copy()
        n = len(tokens_np)
        pos = np.arange(n)

        if max_shifts > 0 and max_shifts < keylen:
            shifts = np.random.choice(keylen, size=max_shifts, replace=False)
        else:
            shifts = np.arange(keylen)

        shift_idx = (shifts[:, None] + pos[None, :]) % keylen
        scores = dp_levenshtein_lazy(tokens_np, xi_np, shift_idx, gamma)
        return float(scores.min())

    def detect(self, tokenized_text, n_runs=100, max_shifts=0, n_workers=0,
               p_threshold=0.05):
        """Permutation-test detection. Returns a dict with the p-value only.

        Returns:
            {"num_tokens_scored", "p_value", "prediction"}.
        """
        tokens_np, xi_np = self._to_numpy(tokenized_text)
        n = len(tokens_np)

        test_result = self.calculate(tokenized_text, max_shifts=max_shifts)

        workers = n_workers if n_workers > 0 else _num_workers()
        workers = min(workers, n_runs)
        runs_per_worker = [n_runs // workers] * workers
        for i in range(n_runs % workers):
            runs_per_worker[i] += 1

        base_seed = int.from_bytes(os.urandom(4), "big")
        args_list = [
            (tokens_np, self.keylen, self.vocab_size, max_shifts, n,
             runs_per_worker[w], test_result, base_seed + w)
            for w in range(workers)
        ]

        p_val = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for count in pool.map(_kth_detect_null_worker, args_list):
                p_val += count

        p_value = (p_val + 1.0) / (n_runs + 1.0)
        return {
            "num_tokens_scored": float(n),
            "p_value": float(p_value),
            "prediction": bool(p_value < p_threshold),
        }
