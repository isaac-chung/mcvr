"""Paired comparisons with cluster resampling.

Queries in the same cluster (for example from one source video) are resampled together in
the bootstrap and sign-flipped together in the permutation test.
"""

from __future__ import annotations

import numpy as np


def _cluster_sums(values: np.ndarray, clusters: list[str]) -> tuple[np.ndarray, np.ndarray]:
    labels = {c: i for i, c in enumerate(dict.fromkeys(clusters))}
    sums, counts = np.zeros(len(labels)), np.zeros(len(labels))
    for v, c in zip(values, clusters):
        sums[labels[c]] += v
        counts[labels[c]] += 1
    return sums, counts


def bootstrap_ci(
    values: np.ndarray, clusters: list[str], n: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    sums, counts = _cluster_sums(values, clusters)
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def permutation_p(values: np.ndarray, clusters: list[str], n: int = 20000, seed: int = 0) -> float:
    """Two-sided p for 'mean difference is zero' by flipping the sign of whole clusters."""
    sums, counts = _cluster_sums(values, clusters)
    observed = abs(sums.sum() / counts.sum())
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n, len(sums)))
    perm = np.abs((signs * sums).sum(axis=1) / counts.sum())
    return float((1 + (perm >= observed - 1e-12).sum()) / (n + 1))


def paired_summary(
    a: dict[str, float], b: dict[str, float], cluster_of: dict[str, str], seed: int = 0
) -> dict[str, float]:
    """Mean of a - b over the shared queries, its sd, CI and p-value."""
    ids = sorted(set(a) & set(b))
    diffs = np.array([a[q] - b[q] for q in ids])
    clusters = [cluster_of[q] for q in ids]
    lo, hi = bootstrap_ci(diffs, clusters, seed=seed)
    return {
        "n": len(ids),
        "mean": float(diffs.mean()),
        "sd": float(diffs.std(ddof=1)) if len(ids) > 1 else 0.0,
        "ci_lo": lo,
        "ci_hi": hi,
        "p": permutation_p(diffs, clusters, seed=seed),
    }
