"""Asset-independent 250-observation PLA thresholds and sampling statistics."""

import numpy as np
from scipy.stats import ks_2samp, spearmanr


def zone(spearman, ks):
    if not np.isfinite(spearman) or not np.isfinite(ks):
        return "undefined"
    if spearman > 0.80 and ks < 0.09:
        return "green"
    if spearman < 0.70 or ks > 0.12:
        return "red"
    return "amber"


def pla(hpl, rtpl):
    if len(hpl) != 250 or len(rtpl) != 250 or not np.all(np.isfinite(np.r_[hpl, rtpl])):
        raise ValueError("PLA requires 250 consecutive finite observations.")
    rho = float(spearmanr(hpl, rtpl).statistic)
    # The two 250-observation empirical CDFs have steps of exactly 1/250.
    ks = round(250 * float(ks_2samp(hpl, rtpl, method="asymp").statistic)) / 250
    return rho, ks, zone(rho, ks)
