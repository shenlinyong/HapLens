"""Null mixed model, whitening transforms, flexible covariate/random-effect builders, and haplotype-class scan.

Model per trait:  y = X b + g + sum_r u_r + e
where g ~ N(0, s_g K) (genomic relationship matrix K), optional group random effects
u_r ~ N(0, s_r Z_r Z_r') (e.g. dam, litter, pen), and e ~ N(0, s_e I).
"""

import os
import numpy as np
import pandas as pd
from scipy import optimize, linalg, stats


def reml_fit(y, X, mats):
    """mats: list of PSD (n,n) matrices (the residual I is added). Returns variance components (len(mats)+1) and V."""
    n = len(y)
    m = len(mats)
    vy = max(float(y.var()), 1e-12)

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
        ld = 2.0 * np.log(np.diag(L)).sum()
        VX = linalg.cho_solve((L, True), X)
        Vy = linalg.cho_solve((L, True), y)
        XtVX = X.T @ VX
        sign, logdet_x = np.linalg.slogdet(XtVX)
        if sign <= 0:
            return 1e10
        b = np.linalg.solve(XtVX, X.T @ Vy)
        r = y - X @ b
        rVr = float(r @ linalg.cho_solve((L, True), r))
        return 0.5 * (ld + logdet_x + rVr)

    th0 = np.log(np.full(m + 1, vy / (m + 1)))
    res = optimize.minimize(
        nll, th0, method="L-BFGS-B", bounds=[(np.log(vy) - 12.0, np.log(vy) + 2.0)] * (m + 1)
    )
    comp = np.exp(res.x)
    return comp, build(res.x)


def whiten(V, method="cholesky"):
    """Return whitening matrix W such that W V W' = I.

    method='cholesky': W = L^-1 where V = L L' (fast, default).
    method='sym':      W = V^{-1/2} (symmetric, permutation/order-invariant).
    """
    if method == "sym":
        evals, evecs = np.linalg.eigh(V)
        evals = np.maximum(evals, 1e-12)
        return (evecs / np.sqrt(evals)) @ evecs.T
    L = linalg.cholesky(V, lower=True)
    return linalg.solve_triangular(L, np.eye(len(V)), lower=True)


def build_design_matrix(df, trait, covars=None, cat_covars=None, no_sex_traits=None):
    """Build fixed-effect design matrix X (with intercept) from phenotype DataFrame.

    When covars and cat_covars are both None, falls back to the classic HapLens convention:
    sex (numeric, omitted if trait in no_sex_traits) + batch (categorical indicators) if present.
    """
    n = len(df)
    no_sex = set(no_sex_traits or ())
    cols = [np.ones(n, dtype=np.float64)]
    if covars is None and cat_covars is None:
        if "sex" in df.columns and trait not in no_sex:
            v = df["sex"].values.astype(float)
            if np.nanstd(v) > 0:
                cols.append(v)
        if "batch" in df.columns:
            vals = df["batch"].values
            uniq = sorted(pd.unique(vals))
            for b in uniq[1:]:
                cols.append((vals == b).astype(np.float64))
    else:
        for c in (covars or ()):
            if c in df.columns and not (c == "sex" and trait in no_sex):
                v = df[c].values.astype(float)
                if np.nanstd(v) > 0:
                    cols.append(v)
        for c in (cat_covars or ()):
            if c in df.columns and not (c == "sex" and trait in no_sex):
                vals = df[c].values
                uniq = sorted(pd.unique(vals))
                for lvl in uniq[1:]:
                    cols.append((vals == lvl).astype(np.float64))
    X = np.column_stack(cols)
    # Drop any collinear columns gracefully via QR/rank check if needed
    return X


def build_random_matrices(df, R, rpos, random_spec="grm,dam", jitter=0.001):
    """Build list of PSD random-effect covariance matrices and their names."""
    ids = df.index.values
    n = len(ids)
    tokens = [x.strip() for x in random_spec.split(",") if x.strip()]
    mats, names = [], []
    for tok in tokens:
        if tok.lower() == "grm":
            K = R[np.ix_([rpos[i] for i in ids], [rpos[i] for i in ids])] + jitter * np.eye(n)
            mats.append(K)
            names.append("grm")
        elif tok in df.columns:
            vals = df[tok].values
            M = (vals[:, None] == vals[None, :]).astype(np.float64)
            mats.append(M)
            names.append(tok)
    return mats, names


def load_grm(rel_prefix):
    """Load PLINK square GRM (.rel and .rel.id). Returns (R, rpos_dict, rid_array)."""
    rid = pd.read_csv(rel_prefix + ".rel.id", sep=r"\s+", header=None, dtype=str)[1].values
    R = np.loadtxt(rel_prefix + ".rel")
    rpos = {i: k for k, i in enumerate(rid)}
    return R, rpos, rid


def resolve_loco_grm(loco_pattern, chrom, fallback_rel):
    """Resolve chromosome-specific LOCO GRM prefix, falling back to full GRM if missing."""
    if not loco_pattern:
        return fallback_rel
    cand = loco_pattern.format(chr=chrom) if "{chr}" in loco_pattern else f"{loco_pattern}_chr{chrom}"
    if os.path.exists(cand + ".rel") and os.path.exists(cand + ".rel.id"):
        return cand
    return fallback_rel


def scan(yt, Xt, W, labels, block=150, want_effects=True):
    """labels: (G, n) int16 class index per individual (-1 excluded). yt, Xt already whitened."""
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
            d = dict(df=q, F=F, P=p, pve=1.0 - rss1 / rss0)
            if want_effects:
                cov = np.linalg.pinv(A1.T @ A1) * (rss1 / dfd)
                k0 = Xt.shape[1] + ne
                d["eff"] = [
                    (int(c), float(c1[k0 + j]), float(np.sqrt(max(cov[k0 + j, k0 + j], 0.0))))
                    for j, c in enumerate(others)
                ]
                d["ref"] = int(ref)
            out.append(d)
    return out
