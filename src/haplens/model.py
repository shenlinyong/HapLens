"""Null mixed model and haplotype-class scan.

Model per trait:  y = X b + g + d + e  with  g ~ N(0, s_g K)  (genomic relationship matrix K),
d ~ N(0, s_d D D')  (dam or other common-environment groups) and  e ~ N(0, s_e I).
The variance components are estimated once under the null by REML. Each haplotype unit is then
tested by a generalised least squares F-test of its dummy-coded classes (reference = most frequent
class; individuals whose class was dropped enter as their own nuisance class).
"""

import numpy as np
from scipy import optimize, linalg, stats


def reml_fit(y, X, mats):
    """mats: list of PSD (n,n) matrices (the residual I is added). Returns variance components (len(mats)+1) and V."""
    n = len(y)
    m = len(mats)
    vy = y.var()

    def build(th):
        V = np.exp(th[-1]) * np.eye(n)
        for k in range(m):
            V += np.exp(th[k]) * mats[k]
        return V

    def nll(th):
        V = build(th)
        try:
            L = linalg.cholesky(V, lower=True)
        except linalg.LinAlgError:
            return 1e10
        ld = 2 * np.log(np.diag(L)).sum()
        VX = linalg.cho_solve((L, True), X)
        Vy = linalg.cho_solve((L, True), y)
        XtVX = X.T @ VX
        b = np.linalg.solve(XtVX, X.T @ Vy)
        r = y - X @ b
        rVr = r @ linalg.cho_solve((L, True), r)
        return 0.5 * (ld + np.linalg.slogdet(XtVX)[1] + rVr)

    th0 = np.log(np.full(m + 1, vy / (m + 1)))
    res = optimize.minimize(nll, th0, method="L-BFGS-B", bounds=[(np.log(vy) - 12, np.log(vy) + 2)] * (m + 1))
    comp = np.exp(res.x)
    return comp, build(res.x)


def whiten(V):
    L = linalg.cholesky(V, lower=True)
    return linalg.solve_triangular(L, np.eye(len(V)), lower=True)  # W = L^-1


def scan(yt, Xt, W, labels, block=150, want_effects=True):
    """labels: (G, n) int16 FH class per individual (-1 excluded). yt, Xt already whitened. Returns list of dict."""
    G, n = labels.shape
    out = []
    for b0 in range(0, G, block):
        blk = labels[b0 : b0 + block]
        cols = []
        spec = []
        for gi in range(len(blk)):
            f = blk[gi]
            present, cnt = np.unique(f[f >= 0], return_counts=True)
            if len(present) < 2:
                spec.append(None)
                continue
            ref = present[np.argmax(cnt)]
            others = [c for c in present if c != ref]
            Z = np.stack([(f == c).astype(np.float64) for c in others], 1)
            ex = (f < 0).astype(np.float64)[:, None] if (f < 0).any() else np.zeros((n, 0))
            spec.append((ref, others, Z.shape[1], ex.shape[1], len(cols)))
            cols.append(np.hstack([Z, ex]))
        if not cols:
            out += [None] * len(blk)
            continue
        Zall = np.hstack(cols)
        WZ = W @ Zall
        pos = 0
        for gi in range(len(blk)):
            s = spec[gi]
            if s is None:
                out.append(None)
                continue
            ref, others, q, ne, _ = s
            Zt = WZ[:, pos : pos + q + ne]
            pos += q + ne
            A0 = np.hstack([Xt, Zt[:, q:]])
            A1 = np.hstack([A0, Zt[:, :q]])
            c0, r0, *_ = np.linalg.lstsq(A0, yt, rcond=None)
            rss0 = float(((yt - A0 @ c0) ** 2).sum())
            c1, r1, rk1, _ = np.linalg.lstsq(A1, yt, rcond=None)
            res1 = yt - A1 @ c1
            rss1 = float((res1**2).sum())
            dfd = n - rk1
            if dfd <= 0 or rss1 <= 0:
                out.append(None)
                continue
            F = ((rss0 - rss1) / q) / (rss1 / dfd)
            p = float(stats.f.sf(F, q, dfd))
            d = dict(df=q, F=F, P=p, pve=1 - rss1 / rss0)
            if want_effects:
                cov = np.linalg.pinv(A1.T @ A1) * (rss1 / dfd)
                k0 = Xt.shape[1] + ne
                d["eff"] = [
                    (int(c), float(c1[k0 + j]), float(np.sqrt(cov[k0 + j, k0 + j])))
                    for j, c in enumerate(others)
                ]
                d["ref"] = int(ref)
            out.append(d)
    return out
