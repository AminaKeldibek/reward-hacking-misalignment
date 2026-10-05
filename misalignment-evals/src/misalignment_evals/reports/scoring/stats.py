"""Binomial summaries for a per-step series of binary scores.

A training step holds one group of rollouts generated from the same weights and the same prompt,
so the scores inside it are independent draws and a plain binomial interval is the right one.
"""
import math
from typing import Optional

import pandas as pd

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[Optional[float], Optional[float]]:
    """Score interval for a proportion.

    Used instead of the normal approximation because it stays inside [0, 1] and holds its
    coverage when the rate is near 0 or 1, which is where most judge scores sit.
    """
    if n <= 0:
        return (None, None)
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def interval_half_width(p: float, n: int, z: float = Z95) -> float:
    """Half-width of the interval on a single step's rate."""
    return z * math.sqrt(p * (1 - p) / n) if n > 0 else float("nan")


def detectable_gap(p: float, n: int, z: float = Z95) -> float:
    """Smallest difference between two steps of this size that clears a two-sided 95% test."""
    return z * math.sqrt(2 * p * (1 - p) / n) if n > 0 else float("nan")


def rate_by_step(frame: pd.DataFrame, step_column: str, value_column: str) -> list[dict]:
    """One point per step: the count, the rate, and its interval."""
    grouped = frame.groupby(step_column)[value_column].agg(k="sum", n="size")
    points = []
    for step, row in grouped.iterrows():
        low, high = wilson(int(row.k), int(row.n))
        points.append({"step": int(step), "k": int(row.k), "n": int(row.n),
                       "rate": float(row.k / row.n), "lo": low, "hi": high})
    return points


def split_by_step(frame: pd.DataFrame, step_column: str, value_column: str,
                  split_column: str, minimum_per_arm: int) -> list[dict]:
    """Per-step rate for each side of a binary column.

    A step is only usable when both arms reach `minimum_per_arm`; below that a rate of 0% or
    100% carries no information, so the point is marked rather than silently drawn.
    """
    levels = sorted(frame[split_column].dropna().unique())
    if len(levels) != 2:
        raise ValueError(
            f"split_column {split_column!r} needs exactly two values, found {list(levels)}")
    low_level, high_level = levels

    rows = []
    for step, group in frame.groupby(step_column):
        arms = [group[group[split_column] == level] for level in (low_level, high_level)]
        row = {"step": int(step),
               "usable": all(len(arm) >= minimum_per_arm for arm in arms)}
        for index, arm in enumerate(arms):
            rate = float(arm[value_column].mean()) if len(arm) else None
            low, high = wilson(int(arm[value_column].sum()), len(arm)) if len(arm) else (None, None)
            row |= {f"n{index}": len(arm), f"rate{index}": rate,
                    f"lo{index}": low, f"hi{index}": high}
        rows.append(row)
    return rows


def reliability(base_rates: list[float], n: int) -> list[dict]:
    """What a step of this size can resolve, for a few plausible base rates."""
    return [{"p": p,
             "half_width": interval_half_width(p, n),
             "gap": detectable_gap(p, n)} for p in base_rates]
