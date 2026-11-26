"""Least-squares continuation regression for simulated stopping policies."""

import numpy as np


def fit_continuation(design, cash):
    return np.linalg.lstsq(design, cash, rcond=1e-8)[0]
