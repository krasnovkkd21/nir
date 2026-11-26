"""Рассчитать профиль интернет-покупок клиента на момент операции."""
from __future__ import annotations
import numpy as np


def category_groups(codes):
    known = np.flatnonzero(codes >= 0)
    if len(known):
        order = known[np.argsort(codes[known], kind="stable")]
        edges = np.flatnonzero(codes[order[1:]] != codes[order[:-1]]) + 1
        yield from np.split(order, edges)
