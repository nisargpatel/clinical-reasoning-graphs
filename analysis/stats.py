from __future__ import annotations
"""Statistical tests for clinical reasoning graph analysis."""

import numpy as np
from scipy.stats import (
    mannwhitneyu,
    wilcoxon,
    kruskal,
    spearmanr,
    pearsonr,
    bootstrap,
)


def permutation_test(
    group_a: list[float],
    group_b: list[float],
    n_permutations: int = 10000,
    alternative: str = "greater",
) -> dict:
    """Permutation test for difference in means.
    
    More robust than parametric tests for small samples and
    non-normal distributions typical in graph similarity scores.
    """
    observed_diff = np.mean(group_a) - np.mean(group_b)
    combined = np.array(group_a + group_b)
    n_a = len(group_a)

    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
        if alternative == "greater":
            if perm_diff >= observed_diff:
                count += 1
        elif alternative == "less":
            if perm_diff <= observed_diff:
                count += 1
        else:
            if abs(perm_diff) >= abs(observed_diff):
                count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return {
        "observed_difference": float(observed_diff),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "alternative": alternative,
    }


def bootstrap_ci(
    data: list[float],
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
) -> dict:
    """Bootstrap confidence interval for a statistic."""
    data = np.array(data)
    stat_fn = np.mean if statistic == "mean" else np.median

    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_fn(sample))

    alpha = 1 - confidence
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return {
        "statistic": statistic,
        "estimate": float(stat_fn(data)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": confidence,
    }


def bonferroni_correct(p_values: list[float]) -> list[float]:
    """Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def fdr_correct(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[sorted_indices[i]] = sorted_p[i]
        else:
            adjusted[sorted_indices[i]] = min(
                sorted_p[i] * n / (i + 1),
                adjusted[sorted_indices[i + 1]],
            )

    return adjusted.tolist()
