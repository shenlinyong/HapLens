"""Bidirectional Local Background-Conditioned (Bi-LBC) association specificity test for coding genes.

Evaluates whether a gene-level coding association is statistically distinguishable from its flanking
non-coding genetic background under the same whitened mixed model. Bi-LBC is an association-specificity
screening tool rather than a causal-variant proof: when coding variants are tightly collinear with
flanking non-coding variants (high max_r2_code_bg), conditioning attenuates the signal and flags the
locus as statistically indistinguishable from local LD.
"""

import argparse
import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from scipy.special import fdtrc

from haplens.kernel import acat_combine, dual_eigvalsh, pval_wchi2
from haplens.model import (
    build_design_matrix,
    build_random_matrices,
    load_grm,
    reml_fit,
    whiten,
)
from haplens.utils import chrom_list


def _bh(pvals):
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    if n == 0:
        return p
    out = np.full(n, np.nan, dtype=np.float64)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    o = np.argsort(pv)
    r = np.empty(len(pv))
    r[o] = np.arange(1, len(pv) + 1)
    q = pv * len(pv) / r
    qs = np.minimum.accumulate(q[o][::-1])[::-1]
    res = np.empty(len(pv))
    res[o] = np.minimum(qs, 1.0)
    out[ok] = res
    return out


def _standardize_geno(G):
    """G: (m, n) int8 dosage (-1 missing) -> standardized (n, m) float64 matrix."""
    X = np.where(G >= 0, G, np.nan).astype(np.float64)
    mu = np.nanmean(X, axis=1, keepdims=True)
    X = np.where(np.isnan(X), mu, X) - mu
    sd = X.std(axis=1)
    keep = sd > 1e-8
    if not keep.any():
        return np.zeros((G.shape[1], 0), dtype=np.float64)
    return (X[keep] / sd[keep, None]).T


def _kernel_score_test(Z_mat, W, Xt_cov, yt):
    """Exact 1-df score test of Z_mat given whitened covariates Xt_cov and whitened yt."""
    if Z_mat.shape[1] == 0:
        return 1.0, 0.0, 1.0
    Q_cov, R_cov = np.linalg.qr(Xt_cov, mode="reduced")
    rk = int(np.sum(np.abs(np.diag(R_cov)) > 1e-10))
    Q_cov = Q_cov[:, :rk]
    n = len(yt)
    dof = max(n - rk, 1)
    r = yt - Q_cov @ (Q_cov.T @ yt)
    rss0 = float(r @ r)
    s2 = rss0 / dof
    Zt = (W @ Z_mat.astype(np.float32)).astype(np.float64)
    B = Zt - Q_cov @ (Q_cov.T @ Zt)
    lam = dual_eigvalsh(B)
    btr = B.T @ r
    q = float((btr**2).sum()) / max(s2, 1e-12)
    pk = pval_wchi2(q, lam)

    btb = np.maximum((B**2).sum(0), 1e-12)
    df1 = max(dof - 1, 1)
    rss1 = np.maximum(rss0 - (btr**2) / btb, 1e-12)
    f_s = ((rss0 - rss1) / rss1) * df1
    p_snps = fdtrc(1, df1, np.maximum(f_s, 0.0))
    p_acatv = acat_combine(p_snps)
    return pk, q, p_acatv


def _top_pcs(Z_mat, W, Q_base, var_frac=0.85, max_pcs=10):
    """Extract orthonormal top PCs of whitened & covariate-projected Z_mat."""
    if Z_mat.shape[1] == 0:
        return np.zeros((Z_mat.shape[0], 0), dtype=np.float64), 0
    Zt = (W @ Z_mat.astype(np.float32)).astype(np.float64)
    B = Zt - Q_base @ (Q_base.T @ Zt)
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    s2 = s**2
    tot = float(s2.sum())
    if tot <= 1e-12:
        return np.zeros((Z_mat.shape[0], 0), dtype=np.float64), 0
    cum = np.cumsum(s2) / tot
    k = int(np.searchsorted(cum, var_frac) + 1)
    k = min(k, max_pcs, int(np.sum(s > 1e-8)))
    if k <= 0:
        return np.zeros((Z_mat.shape[0], 0), dtype=np.float64), 0
    return U[:, :k], k


def bi_lbc_evaluate(
    Z_code,
    Z_bg,
    D,
    bg_var=0.85,
    max_bg_pcs=10,
    n_ctrl=100,
    code_mafs=None,
    bg_mafs=None,
    rng=None,
    alpha_screen=0.01,
    alpha_cond=0.01,
    collinear_r2=0.85,
):
    """Run Bidirectional LBC specificity screening on one gene under pre-whitened trait dict D."""
    W = D["W"]
    Xt = D["Xt"]
    yt = D["yt"]
    Q0 = D["Q0"]
    n = Z_code.shape[0]

    p_code, _, p_code_acatv = _kernel_score_test(Z_code, W, Xt, yt)
    p_code_omni = acat_combine([p_code, p_code_acatv])
    p_bg, _, _ = _kernel_score_test(Z_bg, W, Xt, yt)

    max_r2_cb = 0.0
    if Z_code.shape[1] > 0 and Z_bg.shape[1] > 0:
        max_r2_cb = float((((Z_bg.T @ Z_code) / max(n, 1)) ** 2).max())

    U_bg, k_bg = _top_pcs(Z_bg, W, Q0, var_frac=bg_var, max_pcs=max_bg_pcs)
    bg_r2_exp = 0.0
    if k_bg > 0:
        Zt_c = (W @ Z_code.astype(np.float32)).astype(np.float64)
        Bc = Zt_c - Q0 @ (Q0.T @ Zt_c)
        tot_c = np.maximum((Bc**2).sum(0), 1e-12)
        proj_c = ((U_bg.T @ Bc) ** 2).sum(0)
        bg_r2_exp = float(np.clip(np.mean(proj_c / tot_c), 0.0, 1.0))

        Xt_bg = np.hstack([Xt, U_bg])
        p_lbc, _, p_lbc_acatv = _kernel_score_test(Z_code, W, Xt_bg, yt)
    else:
        p_lbc, p_lbc_acatv = p_code, p_code_acatv

    U_code, k_code = _top_pcs(Z_code, W, Q0, var_frac=0.95, max_pcs=min(max_bg_pcs, Z_code.shape[1]))
    if k_code > 0 and Z_bg.shape[1] > 0:
        Xt_code = np.hstack([Xt, U_code])
        p_rev, _, _ = _kernel_score_test(Z_bg, W, Xt_code, yt)
    else:
        p_rev = p_bg

    p_ctrl_med = np.nan
    p_ctrl_q05 = np.nan
    p_emp_spec = np.nan
    if n_ctrl > 0 and Z_bg.shape[1] >= Z_code.shape[1] and code_mafs is not None and bg_mafs is not None and rng is not None:
        ctrl_ps = []
        m_c = Z_code.shape[1]
        m_b = Z_bg.shape[1]
        for _ in range(n_ctrl):
            used = np.zeros(m_b, dtype=bool)
            picks = []
            for mf in code_mafs:
                cand = np.nonzero(~used & (np.abs(bg_mafs - mf) <= 0.02))[0]
                if len(cand) == 0:
                    cand = np.nonzero(~used & (np.abs(bg_mafs - mf) <= 0.05))[0]
                if len(cand) == 0:
                    cand = np.nonzero(~used)[0]
                if len(cand) == 0:
                    break
                j = int(cand[rng.integers(len(cand))])
                used[j] = True
                picks.append(j)
            if len(picks) == m_c:
                pc_draw, _, _ = _kernel_score_test(Z_bg[:, picks], W, Xt, yt)
                ctrl_ps.append(pc_draw)
        if ctrl_ps:
            ctrl_arr = np.asarray(ctrl_ps, dtype=np.float64)
            p_ctrl_med = float(np.median(ctrl_arr))
            p_ctrl_q05 = float(np.quantile(ctrl_arr, 0.05))
            p_emp_spec = float((1 + (ctrl_arr <= p_code).sum()) / (len(ctrl_arr) + 1))

    lp_code = -np.log10(max(p_code, 1e-300))
    lp_lbc = -np.log10(max(p_lbc, 1e-300))
    csi = float(np.clip(lp_lbc / max(lp_code, 1e-6), 0.0, 1.0)) if lp_code > 0.5 else 0.0

    if p_code >= alpha_screen:
        status = "non_significant"
    elif p_lbc < alpha_cond and (p_rev >= 0.05 or p_lbc < p_rev):
        status = "coding_retained"
    elif p_lbc >= alpha_cond and (max_r2_cb >= collinear_r2 or bg_r2_exp >= collinear_r2):
        status = "collinear_indistinguishable"
    elif p_lbc >= alpha_cond and p_rev < alpha_cond:
        status = "bg_attenuated"
    elif p_lbc >= alpha_cond and p_rev >= alpha_cond:
        status = "collinear_indistinguishable"
    else:
        status = "both_retained"

    return dict(
        P_code=p_code,
        P_code_acatv=p_code_acatv,
        P_code_omni=p_code_omni,
        P_bg=p_bg,
        P_lbc=p_lbc,
        P_lbc_acatv=p_lbc_acatv,
        P_rev=p_rev,
        max_r2_code_bg=round(max_r2_cb, 4),
        bg_r2_explained=round(bg_r2_exp, 4),
        n_bg_pcs=k_bg,
        n_code_pcs=k_code,
        P_ctrl_med=p_ctrl_med,
        P_ctrl_q05=p_ctrl_q05,
        P_emp_spec=p_emp_spec,
        csi=round(csi, 4),
        status=status,
        verdict=status,
    )


def main(argv=None):
    t0 = time.time()
    tracemalloc.start()
    ap = argparse.ArgumentParser(
        prog="haplens lbc-test",
        description="Bidirectional Local Background-Conditioned (Bi-LBC) association specificity screening.",
    )
    ap.add_argument("--bfile", required=True, help="PLINK binary genotype prefix (.bed/.bim/.fam)")
    ap.add_argument("--ann", required=True, help="SNP annotation table (ann_variants.tsv)")
    ap.add_argument("--genes", required=True, help="gene table (ann_genes.tsv)")
    ap.add_argument("--pheno", required=True, help="phenotype CSV")
    ap.add_argument("--rel", required=True, help="PLINK square GRM prefix (.rel and .rel.id)")
    ap.add_argument("--traits", required=True, help="comma-separated trait names")
    ap.add_argument("--no-sex-traits", default="", help="comma-separated traits without sex covariate")
    ap.add_argument("--covar", default=None, help="comma-separated numeric fixed covariates")
    ap.add_argument("--cat-covar", default=None, help="comma-separated categorical fixed covariates")
    ap.add_argument("--random", default="grm,dam", help="random effects specification")
    ap.add_argument("--whiten", choices=["cholesky", "sym"], default="cholesky")
    ap.add_argument("--tier", default="A", choices=["A", "B"], help="functional variant tier")
    ap.add_argument("--chroms", default="1-38", help="chromosomes analysed")
    ap.add_argument("--regions", default="", help="optional comma-separated regions chr:start-end")
    ap.add_argument("--gene-list", default="", help="optional comma-separated gene_ids or gene names")
    ap.add_argument("--flank", type=int, default=100000, help="flanking window on each side of gene (bp)")
    ap.add_argument("--buffer", type=int, default=5000, help="exclusion buffer outside gene body (bp)")
    ap.add_argument("--bg-var", type=float, default=0.85, help="flanking variance fraction captured by PCs")
    ap.add_argument("--max-bg-pcs", type=int, default=10, help="maximum local background PCs")
    ap.add_argument("--max-bg-snps", type=int, default=400, help="even thinning cap for flanking SNPs")
    ap.add_argument("--n-ctrl", type=int, default=100, help="MAF-matched flanking neutral control draws")
    ap.add_argument("--alpha-screen", type=float, default=0.01, help="screening P threshold for status label")
    ap.add_argument("--alpha-cond", type=float, default=0.01, help="conditional P threshold for status label")
    ap.add_argument("--collinear-r2", type=float, default=0.85, help="r^2 threshold flagging collinearity")
    ap.add_argument("--maf", type=float, default=0.01, help="minimum MAF for coding variants")
    ap.add_argument("--bg-maf", type=float, default=0.05, help="minimum MAF for flanking background SNPs")
    ap.add_argument("--call", type=float, default=0.9, help="minimum call rate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lbc_out", help="output directory")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    no_sex = set(filter(None, a.no_sex_traits.split(",")))
    covars = [c for c in a.covar.split(",") if c] if a.covar is not None else None
    cat_covars = [c for c in a.cat_covar.split(",") if c] if a.cat_covar is not None else None

    fam = pd.read_csv(a.bfile + ".fam", sep=r"\s+", header=None, dtype=str)
    n_all = len(fam)
    bpl = (n_all + 3) // 4
    bim = pd.read_csv(
        a.bfile + ".bim", sep=r"\s+", header=None,
        names=["chr", "id", "cm", "pos", "A1", "A2"], dtype={"chr": str},
    )
    bim["row"] = np.arange(len(bim))
    ph = pd.read_csv(a.pheno, dtype={"IID": str}).set_index("IID")
    keep = fam[1].isin(ph.index).values
    iid = fam[1].values[keep]
    ids_geno = set(iid)
    ipos = {i: k for k, i in enumerate(iid)}
    bed = np.memmap(a.bfile + ".bed", dtype=np.uint8, mode="r", offset=3, shape=(len(bim), bpl))
    SH = np.array([0, 2, 4, 6], dtype=np.uint8)
    LUT = np.array([2, -1, 1, 0], dtype=np.int8)

    def dosage(rows):
        raw = np.asarray(bed[rows])
        return LUT[(raw[:, :, None] >> SH & 3).reshape(len(rows), -1)[:, :n_all][:, keep]]

    R, rpos, _ = load_grm(a.rel)
    traits = a.traits.split(",")
    T = {}
    for t in traits:
        req = [t] + (["sex"] if (covars is None and cat_covars is None and "sex" in ph.columns) else [])
        mask = ph.index.isin(rpos.keys()) & ph.index.isin(ids_geno)
        for c in req:
            mask &= ph[c].notna()
        d = ph[mask]
        ids = d.index.values
        y = d[t].values.astype(float)
        y = (y - y.mean()) / max(y.std(), 1e-12)
        X = build_design_matrix(d, t, covars=covars, cat_covars=cat_covars, no_sex_traits=no_sex)
        mats, _ = build_random_matrices(d, R, rpos, random_spec=a.random)
        comp, V = reml_fit(y, X, mats)
        W = whiten(V, method=a.whiten)
        yt = W @ y
        Xt = W @ X
        Q0, R0 = np.linalg.qr(Xt, mode="reduced")
        rk0 = int(np.sum(np.abs(np.diag(R0)) > 1e-10))
        T[t] = dict(
            rows=np.array([ipos[i] for i in ids]),
            W=W.astype(np.float32),
            Xt=Xt,
            yt=yt,
            Q0=Q0[:, :rk0],
            n=len(ids),
        )
        print(f"LBC null fitted: {t} (n={len(ids)}, var_comp={np.round(comp, 4)})", flush=True)

    ann = pd.read_csv(a.ann, sep="\t", dtype={"chr": str})
    genes_df = pd.read_csv(a.genes, sep="\t", dtype={"chr": str}).set_index("gene_id")
    tiers = ["A"] if a.tier == "A" else ["A", "B"]
    func = ann[ann.tier.isin(tiers)].copy()
    func_rows_all = set(ann[ann.tier.isin(["A", "B"])].row.values)

    CH = set(chrom_list(a.chroms))
    genes_sub = genes_df[genes_df.chr.isin(CH)].copy()
    if a.regions:
        rmask = np.zeros(len(genes_sub), dtype=bool)
        for reg in [x.strip() for x in a.regions.split(",") if x.strip()]:
            c, se = reg.split(":")
            s, e = map(int, se.split("-"))
            rmask |= (genes_sub.chr.values == c) & (genes_sub.end.values >= s) & (genes_sub.start.values <= e)
        genes_sub = genes_sub[rmask]
    if a.gene_list:
        gset = {x.strip() for x in a.gene_list.split(",") if x.strip()}
        genes_sub = genes_sub[genes_sub.index.isin(gset) | genes_sub.gene_name.isin(gset)]

    func = func[func.gene_id.isin(genes_sub.index)].copy()
    print(f"Target genes for Bi-LBC test: {func.gene_id.nunique()}", flush=True)

    results = []
    for c in sorted(func.chr.unique(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0)):
        fc = func[func.chr == c]
        bim_c = bim[bim.chr == c].sort_values("pos")
        c_pos = bim_c.pos.values
        c_row = bim_c.row.values
        c_is_func = np.isin(c_row, list(func_rows_all))

        for gid, grp in fc.groupby("gene_id"):
            gmeta = genes_sub.loc[gid]
            gname = str(gmeta["gene_name"])
            gs, ge = int(gmeta["start"]), int(gmeta["end"])

            c_rows_gene = np.sort(grp.row.unique())
            Gc = dosage(c_rows_gene)
            okc = Gc >= 0
            afc = np.maximum(Gc, 0).sum(1) / np.maximum(2 * okc.sum(1), 1)
            mafc = np.minimum(afc, 1.0 - afc)
            good_c = (okc.mean(1) >= a.call) & (mafc >= a.maf)
            if not good_c.any():
                continue
            Gc = Gc[good_c]
            mafc = mafc[good_c]
            Z_code_all = _standardize_geno(Gc)
            if Z_code_all.shape[1] == 0:
                continue

            i0 = np.searchsorted(c_pos, max(gs - a.flank, 1))
            i1 = np.searchsorted(c_pos, ge + a.flank, side="right")
            loc_pos = c_pos[i0:i1]
            loc_row = c_row[i0:i1]
            loc_fn = c_is_func[i0:i1]
            flank_mask = (~loc_fn) & ((loc_pos < gs - a.buffer) | (loc_pos > ge + a.buffer))
            bg_rows = loc_row[flank_mask]
            if len(bg_rows) == 0:
                bg_rows = loc_row[~loc_fn]
            if len(bg_rows) > a.max_bg_snps:
                sel = np.round(np.linspace(0, len(bg_rows) - 1, a.max_bg_snps)).astype(int)
                bg_rows = bg_rows[sel]

            if len(bg_rows) > 0:
                Gb = dosage(bg_rows)
                okb = Gb >= 0
                afb = np.maximum(Gb, 0).sum(1) / np.maximum(2 * okb.sum(1), 1)
                mafb = np.minimum(afb, 1.0 - afb)
                good_b = (okb.mean(1) >= a.call) & (mafb >= a.bg_maf)
                Gb = Gb[good_b]
                mafb = mafb[good_b]
                Z_bg_all = _standardize_geno(Gb) if len(Gb) else np.zeros((len(iid), 0))
            else:
                mafb = np.array([])
                Z_bg_all = np.zeros((len(iid), 0))

            for t, D in T.items():
                Zc = Z_code_all[D["rows"]]
                Zb = Z_bg_all[D["rows"]]
                out = bi_lbc_evaluate(
                    Zc, Zb, D,
                    bg_var=a.bg_var, max_bg_pcs=a.max_bg_pcs,
                    n_ctrl=a.n_ctrl, code_mafs=mafc, bg_mafs=mafb, rng=rng,
                    alpha_screen=a.alpha_screen, alpha_cond=a.alpha_cond,
                    collinear_r2=a.collinear_r2,
                )
                results.append(
                    dict(
                        trait=t, gene_id=gid, gene=gname, chr=c, start=gs, end=ge,
                        n_code_snps=Zc.shape[1], n_bg_snps=Zb.shape[1], **out,
                    )
                )

    res_df = pd.DataFrame(results)
    if len(res_df):
        res_df["q_code"] = res_df.groupby("trait")["P_code"].transform(_bh)
        res_df["q_lbc"] = res_df.groupby("trait")["P_lbc"].transform(_bh)
        res_df = res_df.sort_values(["trait", "P_code"])
    res_df.to_csv(f"{a.out}/lbc_results.csv", index=False)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"Saved {len(res_df)} gene-trait Bi-LBC tests to {a.out}/lbc_results.csv "
        f"(elapsed {time.time()-t0:.1f}s, peak RAM {peak_bytes/(1024*1024):.1f} MB)",
        flush=True,
    )
    if len(res_df):
        cols = ["gene", "chr", "n_code_snps", "max_r2_code_bg", "P_code", "P_bg", "P_lbc", "P_rev", "P_ctrl_med", "csi", "status"]
        for t in traits:
            print(f"\n== Top Bi-LBC results ({t}) ==")
            print(res_df[res_df.trait == t].head(10)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
