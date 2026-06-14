"""Streaming statistics — a chunked Welford accumulator whose result matches a
single-pass computation. Used by the profiling/preprocessing scripts."""
from __future__ import annotations

import numpy as np


class Welford:
    """Numerically stable streaming mean/variance + min/max + null count."""

    __slots__ = ("n", "mean", "M2", "min", "max", "nnull")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = np.inf
        self.max = -np.inf
        self.nnull = 0

    def update(self, arr, total_len=None):
        arr = np.asarray(arr, dtype="float64")
        finite = arr[np.isfinite(arr)]
        if total_len is not None:
            self.nnull += total_len - finite.size
        if finite.size == 0:
            return
        self.min = min(self.min, float(finite.min()))
        self.max = max(self.max, float(finite.max()))
        bn = finite.size
        bm = float(finite.mean())
        bM2 = float(((finite - bm) ** 2).sum())
        delta = bm - self.mean
        tot = self.n + bn
        self.mean += delta * bn / tot
        self.M2 += bM2 + delta**2 * self.n * bn / tot
        self.n = tot

    @property
    def var(self):
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self):
        return float(np.sqrt(self.var))
