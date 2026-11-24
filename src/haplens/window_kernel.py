"""Regional relatedness-kernel and HapLens-O omnibus score test of windows."""

import argparse
import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import fdtrc

from haplens.kernel import acat_combine, acat_matrix, dual_eigvalsh, pval_wchi2
from haplens.model import (
    build_design_matrix,
    build_random_matrices,
    load_grm,
    reml_fit,
    resolve_loco_grm,
    whiten,
)
from haplens.utils import chrom_list


def _fit_trait_null(ph, t, ids_geno, ipos, R, rpos, a, no_sex, covars, cat_covars, rng):
    req_cols = [t]
    if covars is None and cat_covars is None:
        if "sex" in ph.columns:
            req_cols.append("sex")
    else:
        for c in list(covars or ()) + list(cat_covars or ()):
            if c in ph.columns and not (c == "sex" and t in no_sex):
                req_cols.append(c)
    mask = ph.index.isin(rpos.keys()) & ph.index.isin(ids_geno)
    for c in req_cols:
        mask &= ph[c].notna()
    d = ph[mask]
    ids = d.index.values
    m_ = len(ids)
    y = d[t].values.astype(float)
    y = (y - y.mean()) / max(y.std(), 1e-12)
    X = build_design_matrix(d, t, covars=covars, cat_covars=cat_covars, no_sex_traits=no_sex)
    mats, names = build_random_matrices(d, R, rpos, random_spec=a.random)
    comp, V = reml_fit(y, X, mats)
    W = whiten(V, method=a.whiten)
    yt = W @ y
    Xt = W @ X
    Xpinv = np.linalg.pinv(Xt)
    r0 = yt - Xt @ (Xpinv @ yt)
    dof = m_ - int(np.linalg.matrix_rank(Xt))
    rss0 = float(r0 @ r0)
    s2 = rss0 / max(dof, 1)

    RP_proj = np.empty((m_, 0), dtype=np.float64)
    s2_p = np.empty(0, dtype=np.float64)
    rss0_p = np.empty(0, dtype=np.float64)
    if a.perm > 0:
        if a.strata and a.strata in d.columns:
            strata = pd.Series(np.arange(m_)).groupby(d[a.strata].values).apply(list)
        else:
            strata = [list(range(m_))]
        RP = np.empty((m_, a.perm), dtype=np.float64)
        for k in range(a.perm):
            perm = np.arange(m_)
            for idxs in strata:
                idxs = np.asarray(idxs)
                perm[idxs] = rng.permutation(idxs)
            RP[:, k] = r0[perm]
        # Re-project permuted residuals orthogonal to Xt and compute exact residual variance
        RP_proj = RP - Xt @ (Xpinv @ RP)
        rss0_p = (RP_proj**2).sum(0)
        s2_p = rss0_p / max(dof, 1)

    return dict(
        rows=np.array([ipos[i] for i in ids]),
        W=W.astype(np.float32),
        Xt=Xt,
        Xpinv=Xpinv,
        r0=r0,
        rss0=rss0,
        s2=s2,
        dof=dof,
        RP_proj=RP_proj,
        rss0_p=rss0_p,
        s2_p=s2_p,
        comp=comp,
        names=names,
        n=m_,
    )


def main(argv=None):
    t_start = time.time()
    tracemalloc.start()
    ap = argparse.ArgumentParser(
        prog="haplens test-windows",
        description="Regional relatedness-kernel and HapLens-O omnibus score test of windows.",
    )
    ap.add_argument(
        "--no-sex-traits",
        default="",
        help="comma-separated traits analysed without the sex covariate (e.g. hatch weight)",
    )
    ap.add_argument("--bfile", required=True, help="PLINK binary genotype prefix (.bed/.bim/.fam)")
    ap.add_argument(
        "--windows", required=True, help="window table (PLINK .blocks.det layout: CHR BP1 BP2 KB NSNPS SNPS)"
    )
    ap.add_argument("--pheno", required=True, help="phenotype CSV with IID, covariates, pedigree and traits")
    ap.add_argument("--rel", required=True, help="prefix of the PLINK square GRM (.rel and .rel.id)")
    ap.add_argument(
        "--loco-rel",
        default="",
        help="optional LOCO GRM prefix pattern with {chr}, e.g. 'loco/grm_no_chr{chr}'",
    )
    ap.add_argument("--traits", default=None, help="comma-separated trait names")
    ap.add_argument("--covar", default=None, help="comma-separated numeric fixed covariates (default: sex)")
    ap.add_argument(
        "--cat-covar", default=None, help="comma-separated categorical fixed covariates (default: batch)"
    )
    ap.add_argument(
        "--random", default="grm,dam", help="comma-separated random effects: grm and/or column names (default: grm,dam)"
    )
    ap.add_argument(
        "--strata", default="sire", help="column name for stratified permutations (empty for unstratified)"
    )
    ap.add_argument("--whiten", choices=["cholesky", "sym"], default="cholesky", help="whitening matrix")
    ap.add_argument("--chroms", default="1-38", help="chromosomes analysed, e.g. 1-38 or 1-5,Z")
    ap.add_argument("--maf", type=float, default=0.05, help="minimum minor allele frequency")
    ap.add_argument("--call", type=float, default=0.9, help="minimum call rate")
    ap.add_argument(
        "--max-snp",
        type=int,
        default=0,
        help="maximum SNPs per window (0 = use ALL SNPs deterministically; >0 = random subset)",
    )
    ap.add_argument("--min-snp", type=int, default=3, help="minimum number of SNPs per window")
    ap.add_argument("--perm", type=int, default=10, help="number of stratified permutations")
    ap.add_argument("--seed", type=int, default=5, help="random seed")
    ap.add_argument("--out", default="kern", help="output directory")
    a = ap.parse_args(argv)

    no_sex = set(filter(None, a.no_sex_traits.split(",")))
    covars = [c for c in a.covar.split(",") if c] if a.covar is not None else None
    cat_covars = [c for c in a.cat_covar.split(",") if c] if a.cat_covar is not None else None
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    CH = chrom_list(a.chroms)
    fam = pd.read_csv(a.bfile + ".fam", sep=r"\s+", header=None, dtype=str)
    n_all = len(fam)
    bpl = (n_all + 3) // 4
    bim = pd.read_csv(
        a.bfile + ".bim",
        sep=r"\s+",
        header=None,
        names=["chr", "id", "cm", "pos", "A1", "A2"],
        dtype={"chr": str},
    )
    ph = pd.read_csv(a.pheno, dtype={"IID": str}).set_index("IID")
    keep = fam[1].isin(ph.index).values
    iid = fam[1].values[keep]
    ids_geno = set(iid)
    n = len(iid)
    ipos = {i: k for k, i in enumerate(iid)}
    bed = np.memmap(a.bfile + ".bed", dtype=np.uint8, mode="r", offset=3, shape=(len(bim), bpl))
    SH = np.array([0, 2, 4, 6], dtype=np.uint8)
    LUT = np.array([2, -1, 1, 0], dtype=np.int8)

    def dosage(rows):
        raw = np.asarray(bed[rows])
        return LUT[(raw[:, :, None] >> SH & 3).reshape(len(rows), -1)[:, :n_all][:, keep]]

    meta_cols = {"sex", "batch", "sire", "dam"}
    if covars:
        meta_cols.update(covars)
    if cat_covars:
        meta_cols.update(cat_covars)
    if a.strata:
        meta_cols.add(a.strata)
    for tok in a.random.split(","):
        if tok.strip() and tok.strip().lower() != "grm":
            meta_cols.add(tok.strip())
    traits = a.traits.split(",") if a.traits else [c for c in ph.columns if c not in meta_cols]

    # Cache loaded GRMs & fitted trait null models keyed by GRM prefix
    grm_cache = {}
    null_cache = {}

    def get_nulls(rel_prefix):
        if rel_prefix not in null_cache:
            if rel_prefix not in grm_cache:
                grm_cache[rel_prefix] = load_grm(rel_prefix)
            R, rpos, _ = grm_cache[rel_prefix]
            T_map = {}
            for t in traits:
                D = _fit_trait_null(ph, t, ids_geno, ipos, R, rpos, a, no_sex, covars, cat_covars, rng)
                T_map[t] = D
                print(
                    "trait", t, "rel", os.path.basename(rel_prefix), "n", D["n"],
                    "whiten", a.whiten, "var comp", np.round(D["comp"], 4), flush=True,
                )
            null_cache[rel_prefix] = T_map
        return null_cache[rel_prefix]

    if not a.loco_rel:
        get_nulls(a.rel)

    Wd = pd.read_csv(a.windows, sep=r"\s+", dtype={"CHR": str})
    out = {t: [] for t in traits}
    permP_kern = {t: [] for t in traits}
    permP_omni = {t: [] for t in traits}

    for c in CH:
        ridx = np.nonzero(bim.chr.values == c)[0]
        if len(ridx) == 0:
            continue
        rel_c = resolve_loco_grm(a.loco_rel, c, a.rel)
        T = get_nulls(rel_c)
        ridx = ridx[np.argsort(bim.pos.values[ridx], kind="stable")]
        pos = bim.pos.values[ridx]
        G = np.empty((len(ridx), n), np.int8)
        call = np.empty(len(ridx))
        maf = np.empty(len(ridx))
        for s0 in range(0, len(ridx), 100000):
            s1 = min(s0 + 100000, len(ridx))
            g = dosage(ridx[s0:s1])
            G[s0:s1] = g
            ok_ = g >= 0
            call[s0:s1] = ok_.mean(1)
            af = np.maximum(g, 0).sum(1, dtype=np.int64) / np.maximum(2 * ok_.sum(1), 1)
            maf[s0:s1] = np.minimum(af, 1 - af)
        okv = (call >= a.call) & (maf >= a.maf)
        for w in Wd[Wd.CHR == c].itertuples():
            i0, i1 = (np.searchsorted(pos, w.BP1), np.searchsorted(pos, w.BP2, side="right"))
            idx = np.nonzero(okv[i0:i1])[0] + i0
            if len(idx) < a.min_snp:
                continue
            if a.max_snp > 0 and len(idx) > a.max_snp:
                idx = np.sort(rng.choice(idx, a.max_snp, replace=False))
            X = np.where(G[idx] >= 0, G[idx], np.nan).astype(float)
            mu = np.nanmean(X, 1, keepdims=True)
            X = np.where(np.isnan(X), mu, X) - mu
            sd = X.std(1)
            sd[sd == 0] = 1.0
            Z = (X / sd[:, None]).T
            for t, D in T.items():
                Zr = Z[D["rows"]]
                Zt = (D["W"] @ Zr.astype(np.float32)).astype(np.float64)
                B = Zt - D["Xt"] @ (D["Xpinv"] @ Zt)
                lam = dual_eigvalsh(B)
                btr = B.T @ D["r0"]
                q = float((btr**2).sum()) / D["s2"]
                pk = pval_wchi2(q, lam)

                # Within-window single-SNP GLS F-tests & ACAT-V (zero extra matrix multiplications!)
                btb = np.maximum((B**2).sum(0), 1e-12)
                df_snp = max(D["dof"] - 1, 1)
                rss1 = np.maximum(D["rss0"] - (btr**2) / btb, 1e-12)
                f_snp = ((D["rss0"] - rss1) / rss1) * df_snp
                p_snps = fdtrc(1, df_snp, np.maximum(f_snp, 0.0))
                p_acatv = acat_combine(p_snps)
                p_min = float(p_snps.min())
                p_omni = acat_combine([pk, p_acatv])

                out[t].append((c, w.BP1, w.BP2, len(idx), q, pk, p_acatv, p_min, p_omni))

                if a.perm > 0:
                    btR = B.T @ D["RP_proj"]  # (m_snp, n_perm)
                    qp = (btR**2).sum(0) / D["s2_p"]
                    pk_p = np.array([pval_wchi2(x, lam) for x in qp], dtype=np.float64)
                    # Vectorized ACAT-V for all permutations
                    rss1_p = np.maximum(D["rss0_p"][None, :] - (btR**2) / btb[:, None], 1e-12)
                    f_p = ((D["rss0_p"][None, :] - rss1_p) / rss1_p) * df_snp
                    p_snps_p = fdtrc(1, df_snp, np.maximum(f_p, 0.0))  # (m_snp, n_perm)
                    p_acatv_p = acat_matrix(p_snps_p.T)
                    p_omni_p = acat_matrix(np.column_stack([pk_p, p_acatv_p]))
                    permP_kern[t].append(pk_p)
                    permP_omni[t].append(p_omni_p)
        print("chr", c, "windows so far", len(out[traits[0]]), f"elapsed {time.time()-t_start:.1f}s", flush=True)

    tag = a.chroms.replace(",", "_")
    summ = []
    elapsed = time.time() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = round(peak_bytes / (1024 * 1024), 1)
    lam_ = lambda p: float(np.median(stats.chi2.isf(np.clip(p, 1e-300, 1.0), 1)) / 0.4549364) if len(p) else np.nan

    for t in traits:
        cols = ["chr", "start", "end", "n_snp", "Q", "P", "P_acatv", "P_min_snp", "P_omni"]
        r = pd.DataFrame(out[t], columns=cols)
        r.to_csv(f"{a.out}/{t}_{tag}.csv", index=False)
        bonf = 0.05 / max(len(r), 1)
        tail = {}
        if a.perm > 0 and permP_kern[t]:
            per_k = np.vstack(permP_kern[t])  # (n_win, n_perm)
            per_o = np.vstack(permP_omni[t])
            pp = per_k.ravel()
            po = per_o.ravel()
            np.save(f"{a.out}/{t}_{tag}_permP.npy", pp)
            np.save(f"{a.out}/{t}_{tag}_permP_omni.npy", po)
            tail = dict(
                perm_n=int(len(pp)),
                perm_lambda=lam_(pp),
                perm_p05=float((pp < 0.05).mean()),
                perm_p001=float((pp < 0.001).mean()),
                perm_p0001=float((pp < 1e-4).mean()),
                perm_p00001=float((pp < 1e-5).mean()),
                perm_n_p0001=int((pp < 1e-4).sum()),
                perm_n_p00001=int((pp < 1e-5).sum()),
                perm_n_bonf=int((pp < bonf).sum()),
                perm_p05_min=float((per_k < 0.05).mean(0).min()),
                perm_p05_max=float((per_k < 0.05).mean(0).max()),
                perm_p001_min=float((per_k < 0.001).mean(0).min()),
                perm_p001_max=float((per_k < 0.001).mean(0).max()),
                perm_omni_lambda=lam_(po),
                perm_omni_p05=float((po < 0.05).mean()),
                perm_omni_p001=float((po < 0.001).mean()),
                perm_omni_n_p0001=int((po < 1e-4).sum()),
                perm_omni_n_p00001=int((po < 1e-5).sum()),
                perm_omni_n_bonf=int((po < bonf).sum()),
            )
        summ.append(
            dict(
                trait=t,
                chroms=a.chroms,
                whiten=a.whiten,
                loco=bool(a.loco_rel),
                max_snp=a.max_snp,
                windows=len(r),
                lam_obs=lam_(r.P.values),
                lam_omni=lam_(r.P_omni.values),
                n_p001=int((r.P < 0.001).sum()),
                n_bonf_win=int((r.P < bonf).sum()),
                n_bonf_omni=int((r.P_omni < bonf).sum()),
                elapsed_sec=round(elapsed, 1),
                peak_ram_mb=peak_mb,
                **tail,
            )
        )
        print(summ[-1], flush=True)
    pd.DataFrame(summ).to_csv(f"{a.out}/summary_{'-'.join(traits)[:60]}_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
