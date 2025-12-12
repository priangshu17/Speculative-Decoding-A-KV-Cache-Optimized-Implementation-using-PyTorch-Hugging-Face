"""
Metrics for evaluating speculative decoding.
"""

import numpy as np 
from scipy.stats import entropy

def js_divergence(p, q):
    p = np.asarray(p, dtype = np.float64)
    q = np.asarray(q, dtype = np.float64)
    p = p / (p.sum() + 1e-40)
    q = q / (q.sum() + 1e-40)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def compute_speedup(t_base, t_spec):
    return t_base / t_spec if t_spec > 0 else float("inf")
    