"""Page computations moved from docs/js to Python (PLAN_precompute.md). Pure functions, no I/O.

They reproduce what docs/js/stats.js computed in the browser, on the same Float32 values, so the migrated
page shows the same numbers (checked against tests/js/snapshot_before.json by tests/js/migration.test.mjs).
"""

import math

import numpy as np

FLAG_ON_DRIVEN_BOND = 1
FLAG_ON_FROZEN_BOND = 2
FLAG_IS_DRIVEN_TORSION = 4


def observations(
    values: np.ndarray,
    width: int,
    valid,
    topology: np.ndarray,
    flags: np.ndarray,
    n_conf: np.ndarray,
    source: np.ndarray,
    mol_idx: np.ndarray,
    weighting: str,
):
    """Optimization-series observations of one parameter (browser: stats.observations, series=['opt']).

    values: Float32 shard values (rows = assignment x conformer, width 1 or 3); valid: per-row validity or None.
    Returns (value float64 array of the float32 numbers, weight array, molecule array).
    """
    out_v, out_w, out_m = [], [], []
    row = 0
    for i in range(len(topology)):
        t = topology[i]
        rows = n_conf[t]
        if source[t] == "opt" and not (flags[i] & FLAG_ON_FROZEN_BOND):
            for r in range(row, row + rows):
                if valid is not None and not valid[r]:
                    continue
                for w in range(width):
                    out_v.append(float(values[r * width + w]))
                    out_w.append(1.0 / width)
                    out_m.append(int(mol_idx[t]))
        row += rows
    value, weight, mol = np.array(out_v), np.array(out_w), np.array(out_m, dtype=int)
    if weighting == "molecule" and len(value):
        totals: dict[int, float] = {}
        for m, w in zip(mol, weight):  # same accumulation order as the browser
            totals[m] = totals.get(m, 0.0) + w
        weight = np.array([w / totals[m] for m, w in zip(mol, weight)])
    return value, weight, mol


def bins(
    handler: str, values: np.ndarray, centre: float | None
) -> tuple[float, float, int]:
    """Histogram range as the parameter page used it."""
    if handler in ("ProperTorsions", "ImproperTorsions"):
        return -180.0, 180.0, 72
    mn, mx = min(float(values.min()), centre), max(float(values.max()), centre)
    pad = (mx - mn) * 0.05 or (0.01 if handler == "Bonds" else 1.0)
    return mn - pad, mx + pad, 60


def histogram(
    values: np.ndarray, weights: np.ndarray, lo: float, hi: float, nb: int
) -> list[float]:
    counts = [0.0] * nb
    width = (hi - lo) / nb
    for v, w in zip(values, weights):
        b = math.floor((v - lo) / width)
        if b == nb and v == hi:
            b = nb - 1
        if 0 <= b < nb:
            counts[b] += w
    return counts


def linear_stats(values: np.ndarray, weights: np.ndarray) -> dict | None:
    """Weighted mean, population SD, lower weighted median, min, max (browser: weightedLinearStats)."""
    if not len(values):
        return None
    sw = s = 0.0
    for v, w in zip(values, weights):
        sw += w
        s += w * v
    mean = s / sw
    var = 0.0
    for v, w in zip(values, weights):
        var += w * (v - mean) ** 2
    order = sorted(range(len(values)), key=lambda n: values[n])
    acc, median = 0.0, values[order[0]]
    for n in order:
        acc += weights[n]
        if acc >= sw / 2:
            median = values[n]
            break
    return {
        "mean": mean,
        "std": math.sqrt(var / sw),
        "median": float(median),
        "min": float(values[order[0]]),
        "max": float(values[order[-1]]),
    }


def circular_stats(values_deg: np.ndarray, weights: np.ndarray) -> dict | None:
    """Weighted circular mean, SD sqrt(-2 ln R) and R (browser: weightedCircularStats)."""
    if not len(values_deg):
        return None
    c = s = sw = 0.0
    for v, w in zip(values_deg, weights):
        c += w * math.cos(v / (180 / math.pi))
        s += w * math.sin(v / (180 / math.pi))
        sw += w
    r = min(1.0, math.hypot(c, s) / sw)
    return {
        "circMean": math.atan2(s, c) * (180 / math.pi),
        "circStd": math.sqrt(-2 * math.log(r)) * (180 / math.pi) if r > 0 else None,
        "resultantLength": r,
    }


def fourier_profile(
    periodicity: list[int], phase_deg: list[float], k_effective: list[float]
) -> dict:
    """Energy of one torsion parameter at 1° steps over [-180, 180] and its minima (browser: fourierProfile)."""
    x = [float(p) for p in range(-180, 181)]
    y = [
        sum(
            k * (1 + math.cos((n * phi - ph) / (180 / math.pi)))
            for n, ph, k in zip(periodicity, phase_deg, k_effective)
        )
        for phi in x
    ]
    n = len(x) - 1  # periodic: the last point duplicates the first
    minima = [
        {"phi": x[i], "energy": y[i]}
        for i in range(n)
        if y[i] < y[(i - 1 + n) % n] and y[i] <= y[(i + 1) % n]
    ]
    return {"x0": -180, "step": 1, "y": y, "minima": minima}
