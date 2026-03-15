"""
Statistical analysis module — distribution fitting, confidence intervals,
Monte Carlo simulation for probability estimates.
"""

from __future__ import annotations
import math
import random
from core.agent import AgentEstimate


def bootstrap_confidence_interval(
    estimates: list[AgentEstimate],
    n_samples: int = 1000,
    ci_level: float = 0.95,
) -> dict:
    """
    Bootstrap confidence interval for the swarm probability.
    Resamples agent estimates with replacement to estimate uncertainty.
    """
    probs = [e.probability for e in estimates]
    confs = [e.confidence for e in estimates]

    bootstrapped_means = []
    for _ in range(n_samples):
        sample_indices = [random.randint(0, len(probs) - 1) for _ in range(len(probs))]
        sample_probs = [probs[i] for i in sample_indices]
        sample_confs = [confs[i] for i in sample_indices]
        total_conf = sum(sample_confs)
        if total_conf > 0:
            weighted_mean = sum(p * c for p, c in zip(sample_probs, sample_confs)) / total_conf
        else:
            weighted_mean = sum(sample_probs) / len(sample_probs)
        bootstrapped_means.append(weighted_mean)

    bootstrapped_means.sort()
    alpha = (1 - ci_level) / 2
    lower_idx = int(alpha * n_samples)
    upper_idx = int((1 - alpha) * n_samples) - 1

    return {
        "mean": round(sum(bootstrapped_means) / n_samples, 4),
        "ci_lower": round(bootstrapped_means[lower_idx], 4),
        "ci_upper": round(bootstrapped_means[upper_idx], 4),
        "ci_level": ci_level,
        "ci_width": round(bootstrapped_means[upper_idx] - bootstrapped_means[lower_idx], 4),
        "std_error": round(_std(bootstrapped_means), 4),
    }


def monte_carlo_scenarios(
    estimates: list[AgentEstimate],
    n_simulations: int = 5000,
) -> dict:
    """
    Monte Carlo simulation treating each agent's estimate as a
    distribution (beta distribution parameterized by their probability and confidence).
    """
    results = []

    for _ in range(n_simulations):
        sim_probs = []
        for e in estimates:
            # use beta distribution: higher confidence = tighter distribution
            alpha_param = e.probability * e.confidence * 20 + 1
            beta_param = (1 - e.probability) * e.confidence * 20 + 1
            sample = _beta_sample(alpha_param, beta_param)
            sim_probs.append(sample)

        # weighted average for this simulation
        total_conf = sum(e.confidence for e in estimates)
        weighted = sum(p * e.confidence for p, e in zip(sim_probs, estimates)) / total_conf
        results.append(weighted)

    results.sort()

    # compute percentiles
    percentiles = {}
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        idx = int(pct / 100 * len(results))
        percentiles[f"p{pct}"] = round(results[idx], 4)

    # probability of being above/below key thresholds
    thresholds = {}
    for thresh in [0.25, 0.50, 0.75]:
        above = sum(1 for r in results if r > thresh) / len(results)
        thresholds[f"P(>{thresh:.0%})"] = round(above, 3)

    return {
        "percentiles": percentiles,
        "thresholds": thresholds,
        "mean": round(sum(results) / len(results), 4),
        "std": round(_std(results), 4),
        "skew": round(_skewness(results), 4),
        "n_simulations": n_simulations,
    }


def _beta_sample(alpha: float, beta: float) -> float:
    """Simple beta distribution sample using gamma sampling."""
    alpha = max(0.1, alpha)
    beta = max(0.1, beta)
    x = random.gammavariate(alpha, 1)
    y = random.gammavariate(beta, 1)
    return x / (x + y) if (x + y) > 0 else 0.5


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def _skewness(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return 0
    mean = sum(values) / n
    std = _std(values)
    if std == 0:
        return 0
    return sum(((v - mean) / std) ** 3 for v in values) * n / ((n - 1) * (n - 2))


# ---------------------------------------------------------------------------
# Kernel Density Estimation
# ---------------------------------------------------------------------------

def kernel_density_estimate(
    estimates: list[AgentEstimate],
    bandwidth: float | None = None,
) -> dict:
    """
    Gaussian KDE of agent probabilities.

    Evaluates the kernel density at 50 evenly-spaced points on [0, 1]
    and returns the mode (peak), density curve, bandwidth, and whether
    the distribution is bimodal.

    Args:
        estimates: list of AgentEstimate objects.
        bandwidth: fixed bandwidth; if None, uses Silverman's rule.

    Returns:
        dict with mode, density_values, bandwidth_used, bimodal.
    """
    probs = [e.probability for e in estimates]
    n = len(probs)

    if n == 0:
        return {
            "mode": 0.5,
            "density_values": [],
            "bandwidth_used": 0.0,
            "bimodal": False,
        }

    h = bandwidth if bandwidth is not None else _silverman_bandwidth(probs)
    h = max(h, 1e-6)  # guard against zero bandwidth

    # Evaluate KDE at 50 points from 0 to 1
    n_points = 50
    density_values = []
    for i in range(n_points):
        x = i / (n_points - 1)
        density = sum(_gaussian_kernel(x, xi, h) for xi in probs) / n
        density_values.append({"x": round(x, 4), "density": round(density, 6)})

    # Find mode (highest density point)
    best = max(density_values, key=lambda d: d["density"])
    mode = best["x"]

    # Detect bimodality: count local maxima (peaks)
    densities = [d["density"] for d in density_values]
    peaks = 0
    for i in range(1, len(densities) - 1):
        if densities[i] > densities[i - 1] and densities[i] > densities[i + 1]:
            peaks += 1

    return {
        "mode": mode,
        "density_values": density_values,
        "bandwidth_used": round(h, 6),
        "bimodal": peaks >= 2,
    }


def _gaussian_kernel(x: float, xi: float, h: float) -> float:
    """Evaluate Gaussian kernel K((x - xi) / h) / h."""
    z = (x - xi) / h
    return math.exp(-0.5 * z * z) / (h * math.sqrt(2 * math.pi))


def _silverman_bandwidth(data: list[float]) -> float:
    """
    Silverman's rule of thumb: h = 0.9 * min(std, IQR/1.34) * n^(-1/5).
    """
    n = len(data)
    if n < 2:
        return 0.1

    std = _std(data)
    sorted_data = sorted(data)

    # IQR
    q1_idx = int(0.25 * n)
    q3_idx = int(0.75 * n)
    iqr = sorted_data[q3_idx] - sorted_data[q1_idx]

    spread = min(std, iqr / 1.34) if iqr > 0 else std
    if spread == 0:
        spread = 0.1

    return 0.9 * spread * (n ** (-0.2))


# ---------------------------------------------------------------------------
# MCMC Posterior Estimation (Metropolis-Hastings)
# ---------------------------------------------------------------------------

def mcmc_posterior(
    estimates: list[AgentEstimate],
    n_samples: int = 2000,
    burn_in: int = 500,
) -> dict:
    """
    Metropolis-Hastings MCMC sampling for the posterior probability.

    Likelihood: product of beta-distribution PDFs for each agent
    (parameterized by their probability and confidence).
    Prior: uniform on [0, 1].

    Returns:
        dict with posterior_mean, posterior_median, hdi_lower, hdi_upper
        (95% HDI), effective_sample_size, acceptance_rate,
        posterior_samples (last 100 for display).
    """
    if not estimates:
        return {
            "posterior_mean": 0.5,
            "posterior_median": 0.5,
            "hdi_lower": 0.0,
            "hdi_upper": 1.0,
            "effective_sample_size": 0,
            "acceptance_rate": 0.0,
            "posterior_samples": [],
        }

    total_samples = n_samples + burn_in
    samples = []
    theta = 0.5  # start at center
    log_lik_current = _log_likelihood(theta, estimates)
    accepted = 0
    proposal_std = 0.05

    for i in range(total_samples):
        # Propose new theta from normal centered on current
        theta_proposal = theta + random.gauss(0, proposal_std)

        # Reflect to stay in (0, 1)
        if theta_proposal < 0:
            theta_proposal = -theta_proposal
        if theta_proposal > 1:
            theta_proposal = 2 - theta_proposal
        theta_proposal = max(1e-6, min(1 - 1e-6, theta_proposal))

        log_lik_proposal = _log_likelihood(theta_proposal, estimates)

        # Acceptance ratio (prior is uniform so cancels)
        log_alpha = log_lik_proposal - log_lik_current
        if log_alpha >= 0 or random.random() < math.exp(log_alpha):
            theta = theta_proposal
            log_lik_current = log_lik_proposal
            accepted += 1

        if i >= burn_in:
            samples.append(theta)

    acceptance_rate = accepted / total_samples

    # Posterior statistics
    samples.sort()
    n_s = len(samples)
    posterior_mean = sum(samples) / n_s
    posterior_median = samples[n_s // 2]

    # 95% Highest Density Interval — find shortest interval containing 95%
    cred_n = int(0.95 * n_s)
    if cred_n >= n_s:
        hdi_lower = samples[0]
        hdi_upper = samples[-1]
    else:
        min_width = float("inf")
        best_lo = 0
        for lo in range(n_s - cred_n):
            width = samples[lo + cred_n] - samples[lo]
            if width < min_width:
                min_width = width
                best_lo = lo
        hdi_lower = samples[best_lo]
        hdi_upper = samples[best_lo + cred_n]

    # Effective sample size (using autocorrelation at lag 1)
    if n_s > 1:
        mean_s = posterior_mean
        var_s = sum((s - mean_s) ** 2 for s in samples) / n_s
        if var_s > 0:
            autocorr_1 = sum(
                (samples[i] - mean_s) * (samples[i + 1] - mean_s)
                for i in range(n_s - 1)
            ) / ((n_s - 1) * var_s)
            autocorr_1 = max(-0.99, min(0.99, autocorr_1))
            ess = n_s * (1 - autocorr_1) / (1 + autocorr_1)
            ess = max(1, ess)
        else:
            ess = float(n_s)
    else:
        ess = 1.0

    return {
        "posterior_mean": round(posterior_mean, 4),
        "posterior_median": round(posterior_median, 4),
        "hdi_lower": round(hdi_lower, 4),
        "hdi_upper": round(hdi_upper, 4),
        "effective_sample_size": round(ess, 1),
        "acceptance_rate": round(acceptance_rate, 4),
        "posterior_samples": [round(s, 4) for s in samples[-100:]],
    }


def _log_likelihood(theta: float, estimates: list[AgentEstimate]) -> float:
    """
    Log-likelihood of theta given agent estimates.
    Each agent contributes a beta-distribution likelihood parameterized
    by (probability, confidence).
    """
    log_lik = 0.0
    for e in estimates:
        # Beta parameters from agent's prob and confidence
        concentration = e.confidence * 20 + 2  # higher confidence = tighter
        alpha = e.probability * concentration
        beta_param = (1 - e.probability) * concentration
        alpha = max(0.5, alpha)
        beta_param = max(0.5, beta_param)

        # Log of beta PDF: (alpha-1)*log(theta) + (beta-1)*log(1-theta) - logB(alpha, beta)
        # We skip the normalization constant (it doesn't affect MH ratio)
        t = max(1e-10, min(1 - 1e-10, theta))
        log_lik += (alpha - 1) * math.log(t) + (beta_param - 1) * math.log(1 - t)

    return log_lik
