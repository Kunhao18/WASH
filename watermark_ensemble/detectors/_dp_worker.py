"""Standalone DP functions for multiprocess permutation testing.

All functions operate on numpy arrays (no torch, no CUDA) so they can
run in ProcessPoolExecutor workers without GIL contention.
"""

import numpy as np


def dp_levenshtein_lazy(tokens, xi, shift_idx, gamma=0.0):
    """Batched Levenshtein DP with lazy cost lookup (KTH).

    Args:
        tokens: int array [n] — token ids
        xi: float32 array [keylen, vocab_size]
        shift_idx: int array [B, n] — precomputed (shift + pos) % keylen
        gamma: gap penalty

    Returns:
        float32 array [B] — per-batch DP scores
    """
    B, n = shift_idx.shape
    g = np.float32(gamma)

    j_cost = np.arange(n + 1, dtype=np.float32) * g
    dp_prev = np.tile(j_cost, (B, 1))  # [B, n+1]
    dp_curr = np.empty_like(dp_prev)

    for i in range(1, n + 1):
        dp_curr[:, 0] = np.float32(i) * g
        token_i = tokens[i - 1]
        # Lazy: xi[shift_idx[:, :], token_i] → [B, n]
        cost_vec = np.log1p(-xi[shift_idx, token_i]).astype(np.float32)

        for j in range(1, n + 1):
            a = dp_prev[:, j] + g
            b = dp_curr[:, j - 1] + g
            c = dp_prev[:, j - 1] + cost_vec[:, j - 1]
            dp_curr[:, j] = np.minimum(np.minimum(a, b), c)

        dp_prev, dp_curr = dp_curr, dp_prev

    return dp_prev[:, n]


def dp_levenshtein_cost(cost_matrix, gamma=0.0):
    """Batched Levenshtein DP with precomputed costs (ITS).

    Args:
        cost_matrix: float32 array [B, n]
        gamma: gap penalty

    Returns:
        float32 array [B] — per-batch DP scores
    """
    B, n = cost_matrix.shape
    g = np.float32(gamma)

    j_cost = np.arange(n + 1, dtype=np.float32) * g
    dp_prev = np.tile(j_cost, (B, 1))
    dp_curr = np.empty_like(dp_prev)

    for i in range(1, n + 1):
        dp_curr[:, 0] = np.float32(i) * g
        cost_col = cost_matrix[:, i - 1]

        for j in range(1, n + 1):
            a = dp_prev[:, j] + g
            b = dp_curr[:, j - 1] + g
            c = dp_prev[:, j - 1] + cost_col
            dp_curr[:, j] = np.minimum(np.minimum(a, b), c)

        dp_prev, dp_curr = dp_curr, dp_prev

    return dp_prev[:, n]


# ─── Worker functions for ProcessPoolExecutor ───────────────
# Each null worker draws `runs_in_batch` random keys, scores each over the
# searched shifts, and counts how many null scores are at least as extreme as
# the observed test statistic. The caller turns the total count into a p-value
# via (count + 1) / (n_runs + 1) — matching the MarkLLM EXP / EXPEdit
# permutation test. No z-score is computed.


def _kth_detect_null_worker(args):
    """KTH (EXP) null worker for the permutation test."""
    tokens, keylen, vocab_size, max_shifts, n, runs_in_batch, test_result, seed = args
    rng = np.random.RandomState(seed)
    pos = np.arange(n)
    count = 0
    for _ in range(runs_in_batch):
        xi_alt = rng.rand(keylen, vocab_size).astype(np.float32)
        if max_shifts > 0 and max_shifts < keylen:
            shifts = rng.choice(keylen, size=max_shifts, replace=False)
        else:
            shifts = np.arange(keylen)
        shift_idx = (shifts[:, None] + pos[None, :]) % keylen
        scores = dp_levenshtein_lazy(tokens, xi_alt, shift_idx)
        if float(scores.min()) <= test_result:
            count += 1
    return count


def _its_detect_null_worker(args):
    """ITS (EXPEdit) null worker for the permutation test."""
    token_ranks_np, keylen, max_shifts, n, runs_in_batch, test_result, seed, gamma = args
    rng = np.random.RandomState(seed)
    pos = np.arange(n)
    count = 0
    for _ in range(runs_in_batch):
        xi_alt = rng.rand(keylen, 1).astype(np.float32)
        if max_shifts > 0 and max_shifts < keylen:
            shifts = rng.choice(keylen, size=max_shifts, replace=False)
        else:
            shifts = np.arange(keylen)
        rand_idx = (shifts[:, None] + pos[None, :]) % keylen
        xi_rand = xi_alt[rand_idx].squeeze(-1)
        dist_rand = np.abs(token_ranks_np[None, :] - xi_rand)
        dist_rand = np.clip(dist_rand, 0, 1.0 - 1e-7)
        cost_rand = np.log1p(-dist_rand).astype(np.float32)
        scores = dp_levenshtein_cost(cost_rand, gamma)
        if float(scores.min()) >= test_result:
            count += 1
    return count
