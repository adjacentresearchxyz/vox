"""Utility functions for parsing and normalizing forecasts."""

import re
from typing import Optional
import numpy as np


def parse_probability(text: str) -> Optional[float]:
    match = re.search(r"PROBABILITY:\s*([\d.]+)", text, re.IGNORECASE)
    if match:
        prob = float(match.group(1))
        return normalize_probability(prob)

    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        prob = float(match.group(1)) / 100
        return normalize_probability(prob)

    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:in|out of)\s*(\d+)", text, re.IGNORECASE)
    if match:
        num = float(match.group(1))
        denom = float(match.group(2))
        if denom > 0:
            return normalize_probability(num / denom)

    match = re.search(
        r"(?:odds|chance|likelihood).*?(\d+(?:\.\d+)?)", text, re.IGNORECASE
    )
    if match:
        prob = float(match.group(1))
        if prob > 1:
            prob = prob / 100
        return normalize_probability(prob)

    return None


def normalize_probability(prob: float) -> float:
    return max(0.01, min(0.99, prob))


def weighted_average(probabilities: list[float], weights: list[float]) -> float:
    if not probabilities or not weights:
        return 0.5

    if len(probabilities) != len(weights):
        raise ValueError("Probabilities and weights must have same length")

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.5

    weighted_sum = sum(p * w for p, w in zip(probabilities, weights))
    return weighted_sum / total_weight


def parse_percentiles(text: str) -> dict[int, float]:
    percentiles = {}
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        match = re.search(rf"p{p}:\s*([\d.]+)", text, re.IGNORECASE)
        if match:
            percentiles[p] = float(match.group(1))
    return percentiles


def parse_median(text: str) -> Optional[float]:
    match = re.search(r"MEDIAN:\s*([\d.]+)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def parse_option_probabilities(text: str, options: list[str]) -> dict[str, float]:
    probs = {}
    for option in options:
        pattern = rf"{re.escape(option)}:\s*([\d.]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            probs[option] = float(match.group(1))
    return probs


def extract_reasoning(text: str) -> str:
    match = re.search(
        r"REASONING:\s*(.*?)(?=PROBABILITY:|MEDIAN:|PROBABILITIES:|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def cdf_from_percentiles(
    percentiles: dict[int, float], lower: float, upper: float
) -> list[float]:
    cdf = [0.0] * 201
    sorted_p = sorted(percentiles.items())

    for i in range(201):
        x = lower + (upper - lower) * i / 200

        for j, (p, val) in enumerate(sorted_p):
            if x < val:
                if j == 0:
                    cdf[i] = (
                        p / 100 * (x - lower) / (val - lower) if val != lower else 0
                    )
                else:
                    prev_p, prev_val = sorted_p[j - 1]
                    t = (x - prev_val) / (val - prev_val) if val != prev_val else 0
                    cdf[i] = (prev_p + t * (p - prev_p)) / 100
                break
        else:
            cdf[i] = 1.0

    cdf[0] = 0.0
    cdf[-1] = 1.0

    for i in range(1, 201):
        cdf[i] = max(cdf[i], cdf[i - 1])

    return cdf


def logit(p: float) -> float:
    p = max(0.0001, min(0.9999, p))
    return np.log(p / (1 - p))


def inv_logit(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def aggregate_logit_space(probabilities: list[float], weights: list[float]) -> float:
    logits = [logit(p) for p in probabilities]
    avg_logit = weighted_average(logits, weights)
    return inv_logit(avg_logit)
