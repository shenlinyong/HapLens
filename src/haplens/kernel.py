"""Tail probability of a weighted sum of chi-square(1) variables by the Lugannani-Rice saddlepoint approximation (Kuonen, 1999).

Used for the score statistic of the regional relatedness-kernel test.
"""

import numpy as np
from scipy.stats import norm


def pval_wchi2(q, lam):
    lam = lam[lam > 1e-10 * max(lam.max(), 1e-300)] if len(lam) else lam
    if len(lam) == 0 or q <= 0:
        return 1.0
    mu = lam.sum()
    lmax = lam.max()
    hi = 0.5 / lmax * (1 - 1e-12)
    if abs(q - mu) < 1e-9 * mu:
        return 0.5

    def Kp(t):
        return (lam / (1 - 2 * t * lam)).sum()

    if q > mu:
        lo_, hi_ = 0.0, hi
    else:
        lo_, hi_ = -1.0 / lmax * 1e6, 0.0
    for _ in range(100):  # bisection for K'(t) = q
        mid = 0.5 * (lo_ + hi_)
        if Kp(mid) < q:
            lo_ = mid
        else:
            hi_ = mid
    t = 0.5 * (lo_ + hi_)
    K = -0.5 * np.log(1 - 2 * t * lam).sum()
    K2 = 2 * (lam**2 / (1 - 2 * t * lam) ** 2).sum()
    w = np.sign(t) * np.sqrt(max(2 * (t * q - K), 1e-300))
    v = t * np.sqrt(K2)
    if w == 0 or v == 0:
        return 0.5
    z = w + np.log(v / w) / w
    return float(norm.sf(z))
