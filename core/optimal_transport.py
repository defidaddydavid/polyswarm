"""
Optimal Transport — Wasserstein distance for comparing aggregation methods.

Measures the "earth mover's distance" between probability distributions implied
by different aggregation methods. Clusters methods by agreement, identifies
outliers, and produces a robust consensus from the largest cluster.

References:
  - Kantorovich, L.V. (1942). "On the Translocation of Masses."
    Dokl. Akad. Nauk SSSR.
  - Villani, C. (2009). "Optimal Transport: Old and New." Springer.
  - Peyre, G. & Cuturi, M. (2019). "Computational Optimal Transport."
    Foundations and Trends in ML.
"""

from __future__ import annotations


def method_distance_analysis(method_results: dict[str, float]) -> dict:
    """
    Main entry point. Compute pairwise Wasserstein distances between
    aggregation methods, cluster them, and find a robust consensus.

    Args:
        method_results: dict of method_name -> probability
            e.g. {"weighted": 0.65, "bayesian": 0.72, "monte_carlo": 0.68}

    Returns:
        dict with pairwise_distances, clusters, robust_consensus,
        outlier_methods, method_agreement_score, largest_cluster_size,
        cluster_spread.
    """
    names = list(method_results.keys())
    n = len(names)

    if n == 0:
        return {
            "pairwise_distances": {},
            "clusters": [],
            "robust_consensus": 0.5,
            "outlier_methods": [],
            "method_agreement_score": 0.0,
            "largest_cluster_size": 0,
            "cluster_spread": 0.0,
        }

    if n == 1:
        return {
            "pairwise_distances": {},
            "clusters": [names],
            "robust_consensus": round(method_results[names[0]], 4),
            "outlier_methods": [],
            "method_agreement_score": 1.0,
            "largest_cluster_size": 1,
            "cluster_spread": 0.0,
        }

    # Compute pairwise 1D Wasserstein distances
    pairwise_distances = {}
    distance_matrix = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = _wasserstein_1d(method_results[names[i]], method_results[names[j]])
            key = f"{names[i]}_vs_{names[j]}"
            pairwise_distances[key] = round(d, 4)
            distance_matrix[(names[i], names[j])] = d
            distance_matrix[(names[j], names[i])] = d

    # Determine clustering threshold — adaptive based on spread
    all_probs = list(method_results.values())
    spread = max(all_probs) - min(all_probs)
    # threshold: methods within 1/3 of total spread are clustered together,
    # but at least 0.03 and at most 0.15
    threshold = max(0.03, min(0.15, spread / 3.0)) if spread > 0 else 0.05

    clusters = _single_linkage_cluster(distance_matrix, names, threshold)

    # Find largest cluster
    largest_cluster = max(clusters, key=len)
    largest_cluster_size = len(largest_cluster)

    # Outliers: methods not in the largest cluster
    largest_set = set(largest_cluster)
    outlier_methods = [m for m in names if m not in largest_set]

    # Robust consensus from largest cluster
    robust_consensus = _robust_consensus(clusters, method_results)

    # Cluster spread: std dev of probabilities in the largest cluster
    if largest_cluster_size > 1:
        cluster_probs = [method_results[m] for m in largest_cluster]
        cmean = sum(cluster_probs) / len(cluster_probs)
        cluster_spread = (sum((p - cmean) ** 2 for p in cluster_probs) / len(cluster_probs)) ** 0.5
    else:
        cluster_spread = 0.0

    # Method agreement score: 1 - normalized mean pairwise distance
    if pairwise_distances:
        mean_dist = sum(pairwise_distances.values()) / len(pairwise_distances)
        # normalize: max meaningful distance is 1.0 (prob 0 vs prob 1)
        agreement_score = max(0.0, 1.0 - mean_dist * 2)
    else:
        agreement_score = 1.0

    return {
        "pairwise_distances": pairwise_distances,
        "clusters": [[m for m in c] for c in clusters],
        "robust_consensus": round(robust_consensus, 4),
        "outlier_methods": outlier_methods,
        "method_agreement_score": round(agreement_score, 4),
        "largest_cluster_size": largest_cluster_size,
        "cluster_spread": round(cluster_spread, 4),
    }


def _wasserstein_1d(a: float, b: float) -> float:
    """
    1D Wasserstein distance between two point estimates.
    For point masses, this is simply |a - b|.
    For empirical distributions (lists), sort and compute mean abs diff.
    """
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        sa = sorted(a)
        sb = sorted(b)
        n = min(len(sa), len(sb))
        if n == 0:
            return 0.0
        # interpolate to same length if needed
        if len(sa) != len(sb):
            # resample to same length using linear interpolation
            def _resample(arr, target_len):
                if len(arr) == target_len:
                    return arr
                result = []
                for i in range(target_len):
                    pos = i * (len(arr) - 1) / (target_len - 1) if target_len > 1 else 0
                    lo = int(pos)
                    hi = min(lo + 1, len(arr) - 1)
                    frac = pos - lo
                    result.append(arr[lo] * (1 - frac) + arr[hi] * frac)
                return result

            target = max(len(sa), len(sb))
            sa = _resample(sa, target)
            sb = _resample(sb, target)

        return sum(abs(x - y) for x, y in zip(sa, sb)) / len(sa)

    return abs(a - b)


def _single_linkage_cluster(
    distance_matrix: dict,
    names: list[str],
    threshold: float,
) -> list[list[str]]:
    """
    Single-linkage agglomerative clustering.
    Merges clusters when the minimum distance between any pair of
    elements across two clusters is below the threshold.
    """
    # Start with each method in its own cluster
    clusters: list[set[str]] = [{name} for name in names]

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                # find minimum distance between clusters i and j
                min_dist = float("inf")
                for a in clusters[i]:
                    for b in clusters[j]:
                        if a != b:
                            d = distance_matrix.get((a, b), float("inf"))
                            if d < min_dist:
                                min_dist = d
                if min_dist <= threshold:
                    # merge j into i
                    clusters[i] = clusters[i] | clusters[j]
                    clusters.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1

    # Convert sets to sorted lists for deterministic output
    return [sorted(c) for c in clusters]


def _robust_consensus(
    clusters: list[list[str]],
    method_probs: dict[str, float],
) -> float:
    """
    Compute a robust consensus from the largest cluster using a trimmed mean.
    Trims the most extreme value if cluster has 4+ members.
    """
    # Find largest cluster
    largest = max(clusters, key=len)
    probs = sorted(method_probs[m] for m in largest)

    if len(probs) == 0:
        return 0.5

    if len(probs) <= 3:
        # too few to trim, just use mean
        return sum(probs) / len(probs)

    # Trim top and bottom ~10%
    trim_count = max(1, len(probs) // 10)
    trimmed = probs[trim_count: len(probs) - trim_count]

    if not trimmed:
        return sum(probs) / len(probs)

    return sum(trimmed) / len(trimmed)
