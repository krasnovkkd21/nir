from __future__ import annotations
import numpy as np


def batch_clients(clients):
    bounds = np.r_[0, np.flatnonzero(clients[1:] != clients[:-1]) + 1, len(clients)]
    return clients[bounds[:-1]], bounds


def index_history(clients, timestamps, starts, ends, offset):
    _, bounds = batch_clients(clients)
    for begin, end in zip(bounds[:-1], bounds[1:], strict=True):
        times = timestamps[begin:end]
        if np.any(times[1:] < times[:-1]):
            raise ValueError("Операции клиента записаны вне хронологического порядка.")
        starts[begin:end] = begin + offset
        ends[begin:end] = np.searchsorted(times, times, side="left") + begin + offset
