"""Quadratic-form tail probabilities (numerically stable Kuonen saddlepoint) and ACAT combination."""

import numpy as np
from scipy.stats import norm


def dual_eigvalsh(B):
    """Exact nonzero eigenvalues of B'B using the smaller Gram matrix (min(n, m) x min(n, m))."""
    if B.shape[1] <= B.shape[0]:
        return np.linalg.eigvalsh(B.T @ B)
    return np.linalg.eigvalsh(B @ B.T)


def pval_wchi2(q, lam):
    """Upper-tail P(sum_k lam_k * chi2_1 >= q) via cancellation-free Lugannani-Rice saddlepoint (Kuonen, 1999)."""
    if len(lam) == 0 or q <= 0:
        return 1.0
    lmax_raw = float(lam.max())
    if lmax_raw <= 0:
        return 1.0
    lam = lam[lam > 1e-10 * max(lmax_raw, 1e-300)]
    if len(lam) == 0:
        return 1.0
    mu = float(lam.sum())
    lmax = float(lam.max())
    k2_0 = 2.0 * float((lam**2).sum())
    k3_0 = 8.0 * float((lam**3).sum())
    sd0 = np.sqrt(max(k2_0, 1e-300))

    # Exact continuous Lugannani-Rice limit near q = mu (prevents catastrophic cancellation in t*q - K)
    if abs(q - mu) < 1e-3 * sd0:
        z = (q - mu) / sd0 + k3_0 / (6.0 * sd0**3)
        return float(max(norm.sf(z), 1e-300))

    hi = 0.5 / lmax * (1.0 - 1e-12)

    def Kp(t):
        return float((lam / (1.0 - 2.0 * t * lam)).sum())

    if q > mu:
        lo_, hi_ = 0.0, hi
    else:
        lo_, hi_ = -1.0 / lmax, 0.0
        while Kp(lo_) > q and lo_ > -1e12 / lmax:
            lo_ *= 10.0

    for _ in range(100):
        mid = 0.5 * (lo_ + hi_)
        if Kp(mid) < q:
            lo_ = mid
        else:
            hi_ = mid
    t = 0.5 * (lo_ + hi_)
    u = 2.0 * t * lam

    # Cancellation-free 2*(t*q - K(t)) = sum_k [ u_k/(1-u_k) + log1p(-u_k) ]
    small = np.abs(u) < 1e-3
    term = np.empty_like(u)
    us = u[small]
    term[small] = (us**2) * (0.5 + us * (2.0 / 3.0 + us * (0.75 + us * (0.8 + us * (5.0 / 6.0)))))
    ul = u[~small]
    term[~small] = ul / (1.0 - ul) + np.log1p(-ul)
    val = float(term.sum())

    K2 = 2.0 * float((lam**2 / (1.0 - u) ** 2).sum())
    w = np.sign(t) * np.sqrt(max(val, 1e-300))
    v = t * np.sqrt(max(K2, 1e-300))
    if abs(w) < 1e-6 or abs(v) < 1e-6 or (v / w) <= 0.0:
        z = (q - mu) / sd0 + k3_0 / (6.0 * sd0**3)
        return float(max(norm.sf(z), 1e-300))
    z = w + np.log(v / w) / w
    return float(max(norm.sf(z), 1e-300))


def acat_combine(pvals, weights=None):
    """Aggregated Cauchy Association Test (Liu & Xie, 2020) for a 1D array of p-values."""
    p = np.asarray(pvals, dtype=np.float64)
    ok = np.isfinite(p)
    if not ok.any():
        return 1.0
    p = np.clip(p[ok], 1e-300, 1.0 - 1e-16)
    w = np.full(len(p), 1.0 / len(p)) if weights is None else np.asarray(weights, dtype=np.float64)[ok]
    w = w / w.sum()
    small = p < 1e-15
    t = np.where(small, 1.0 / (np.pi * p), np.tan((0.5 - p) * np.pi))
    T = float((w * t).sum())
    if T > 1e15:
        return float(max(1.0 / (np.pi * T), 1e-300))
    return float(np.clip(0.5 - np.arctan(T) / np.pi, 1e-300, 1.0))


def acat_matrix(P, weights=None):
    """Row-wise ACAT across columns of a 2D array P (n_tests x k)."""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 1:
        return acat_combine(P, weights=weights)
    P_clip = np.clip(P, 1e-300, 1.0 - 1e-16)
    small = P_clip < 1e-15
    Tmat = np.where(small, 1.0 / (np.pi * P_clip), np.tan((0.5 - P_clip) * np.pi))
    T = Tmat.mean(axis=1) if weights is None else Tmat @ (np.asarray(weights, dtype=np.float64) / np.sum(weights))
    return np.where(
        T > 1e15,
        np.maximum(1.0 / (np.pi * np.maximum(T, 1e-300)), 1e-300),
        np.clip(0.5 - np.arctan(T) / np.pi, 1e-300, 1.0),
    )
