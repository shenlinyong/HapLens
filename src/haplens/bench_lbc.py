"""LD-stratified benchmark of Bi-LBC under True Coding vs Flanking Non-Coding Hitchhiking QTLs."""

import argparse
import os
import time
import tracemalloc
import numpy as np
import pandas as pd

from haplens.lbc_test import _standardize_geno, _kernel_score_test, bi_lbc_evaluate
from haplens.model import build_design_matrix, build_random_matrices, load_grm, reml_fit, scan, whiten
from haplens.utils import chrom_list


def _wilson_ci(k, n, z=1.959963984540054):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _diplotype_classes(G, min_n=20):
    n = G.shape[1]
    ok = (G >= 0).all(0)
    lab = np.full(n, -1, dtype=np.int16)
    if ok.sum() < 2 * min_n:
        return lab, 0
    _, inv, cnt = np.unique(G[:, ok].T, axis=0, return_inverse=True, return_counts=True)
    kc = np.nonzero(cnt >= min_n)[0]
    if len(kc) < 2:
        return lab, int(len(kc))
    io = np.nonzero(ok)[0]
    for j, c_ in enumerate(kc[np.argsort(-cnt[kc])]):
        lab[io[inv == c_]] = j
    return lab, int(len(kc))


def main(argv=None):
    t0 = time.time()
    tracemalloc.start()
    ap = argparse.ArgumentParser(
        prog="haplens bench-lbc",
        description="LD-stratified benchmark of Bi-LBC under true coding vs flanking non-coding QTLs.",
    )
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--genes", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--rel", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--no-sex-traits", default="")
    ap.add_argument("--chroms", default="1-38")
    ap.add_argument("--n-genes", type=int, default=150)
    ap.add_argument("--pve", default="0.02,0.04")
    ap.add_argument("--flank", type=int, default=100000)
    ap.add_argument("--buffer", type=int, default=5000)
    ap.add_argument("--min-hitch-r2", type=float, default=0.25)
    ap.add_argument("--max-hitch-r2", type=float, default=0.85)
    ap.add_argument("--bg-var", type=float, default=0.85)
    ap.add_argument("--max-bg-pcs", type=int, default=10)
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", default="bench_lbc.csv")
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    no_sex = set(filter(None, a.no_sex_traits.split(",")))

    fam = pd.read_csv(a.bfile + ".fam", sep=r"\s+", header=None, dtype=str)
    n_all = len(fam)
    bpl = (n_all + 3) // 4
    bim = pd.read_csv(
        a.bfile + ".bim", sep=r"\s+", header=None,
        names=["chr", "id", "cm", "pos", "A1", "A2"], dtype={"chr": str},
    )
    bim["row"] = np.arange(len(bim))
    ph = pd.read_csv(a.pheno, dtype={"IID": str}).set_index("IID")
    R, rpos, _ = load_grm(a.rel)

    t = a.trait
    mask = ph[t].notna() & ph.index.isin(rpos.keys()) & ph.index.isin(fam[1].values)
    if "sex" in ph.columns:
        mask &= ph["sex"].notna()
    d = ph[mask]
    ids = d.index.values
    n = len(ids)
    keep = fam[1].isin(ids).values
    order = pd.Series(np.arange(keep.sum()), index=fam[1].values[keep])
    rowsel = order.loc[ids].values

    bed = np.memmap(a.bfile + ".bed", dtype=np.uint8, mode="r", offset=3, shape=(len(bim), bpl))
    SH = np.array([0, 2, 4, 6], dtype=np.uint8)
    LUT = np.array([2, -1, 1, 0], dtype=np.int8)

    def dosage_sel(rows):
        raw = np.asarray(bed[rows])
        return LUT[(raw[:, :, None] >> SH & 3).reshape(len(rows), -1)[:, :n_all][:, keep]][:, rowsel]

    y0 = d[t].values.astype(float)
    y0 = (y0 - y0.mean()) / max(y0.std(), 1e-12)
    X = build_design_matrix(d, t, no_sex_traits=no_sex)
    mats, _ = build_random_matrices(d, R, rpos, random_spec="grm,dam")
    _, V = reml_fit(y0, X, mats)
    W = whiten(V)
    W32 = W.astype(np.float32)
    Xt = W @ X
    Q0, R0 = np.linalg.qr(Xt, mode="reduced")
    Q0 = Q0[:, : int(np.sum(np.abs(np.diag(R0)) > 1e-10))]

    ann = pd.read_csv(a.ann, sep="\t", dtype={"chr": str})
    genes_df = pd.read_csv(a.genes, sep="\t", dtype={"chr": str}).set_index("gene_id")
    CH = set(chrom_list(a.chroms))
    funcA = ann[(ann.tier == "A") & ann.chr.isin(CH)].copy()
    func_all_rows = set(ann[ann.tier.isin(["A", "B"])].row.values)

    vcnt = funcA.groupby("gene_id")["row"].nunique()
    cand_gids = rng.permutation(vcnt[vcnt >= 2].index.values)

    bim_by_chr = {}
    for c in CH:
        sub = bim[bim.chr == c].sort_values("pos")
        if len(sub):
            bim_by_chr[c] = (sub.pos.values, sub.row.values, np.isin(sub.row.values, list(func_all_rows)))

    pve_list = [float(x) for x in a.pve.split(",")]
    rows_out = []
    done = 0

    for gid in cand_gids:
        if done >= a.n_genes:
            break
        if gid not in genes_df.index:
            continue
        gm = genes_df.loc[gid]
        c = str(gm["chr"])
        if c not in bim_by_chr:
            continue
        gs, ge = int(gm["start"]), int(gm["end"])
        g_rows = np.sort(funcA[funcA.gene_id == gid].row.unique())
        Gc = dosage_sel(g_rows)
        okc = Gc >= 0
        afc = np.maximum(Gc, 0).sum(1) / np.maximum(2 * okc.sum(1), 1)
        mafc = np.minimum(afc, 1.0 - afc)
        good_c = (okc.mean(1) >= 0.9) & (mafc >= 0.05)
        if good_c.sum() < 2:
            continue
        Gc = Gc[good_c]
        mafc = mafc[good_c]
        Z_code = _standardize_geno(Gc)
        if Z_code.shape[1] < 2:
            continue
        lab_c, ncls_c = _diplotype_classes(Gc, min_n=a.min_n)
        if ncls_c < 2:
            continue

        c_pos, c_row, c_fn = bim_by_chr[c]
        i0 = np.searchsorted(c_pos, max(gs - a.flank, 1))
        i1 = np.searchsorted(c_pos, ge + a.flank, side="right")
        lp, lr, lf = c_pos[i0:i1], c_row[i0:i1], c_fn[i0:i1]
        fmask = (~lf) & ((lp < gs - a.buffer) | (lp > ge + a.buffer))
        bg_rows = lr[fmask]
        if len(bg_rows) < 20:
            continue
        if len(bg_rows) > 300:
            bg_rows = bg_rows[np.round(np.linspace(0, len(bg_rows) - 1, 300)).astype(int)]
        Gb = dosage_sel(bg_rows)
        okb = Gb >= 0
        afb = np.maximum(Gb, 0).sum(1) / np.maximum(2 * okb.sum(1), 1)
        mafb = np.minimum(afb, 1.0 - afb)
        good_b = (okb.mean(1) >= 0.9) & (mafb >= 0.05)
        if good_b.sum() < 15:
            continue
        Gb = Gb[good_b]
        mafb = mafb[good_b]
        Z_bg = _standardize_geno(Gb)
        if Z_bg.shape[1] < 15:
            continue

        ld_mat = ((Z_bg.T @ Z_code) / n) ** 2
        max_r2_bg = ld_mat.max(axis=1)
        hitch_any = np.nonzero((max_r2_bg >= a.min_hitch_r2) & (max_r2_bg <= a.max_hitch_r2))[0]
        if len(hitch_any) == 0:
            continue

        used = np.zeros(Z_bg.shape[1], dtype=bool)
        ctrl_picks = []
        for mf in mafc:
            cand = np.nonzero(~used & (np.abs(mafb - mf) <= 0.03))[0]
            if len(cand) == 0:
                cand = np.nonzero(~used)[0]
            j = int(rng.choice(cand))
            used[j] = True
            ctrl_picks.append(j)
        Z_ctrl1 = Z_bg[:, ctrl_picks]
        Xt_ctrl1 = np.hstack([Xt, W @ Z_ctrl1])

        code_idx = int(rng.choice(Z_code.shape[1]))
        code_max_r2 = float(ld_mat[:, code_idx].max())
        g_code = Z_code[:, code_idx]

        hitch_idx = int(rng.choice(hitch_any))
        hitch_r2 = float(max_r2_bg[hitch_idx])
        g_hitch = Z_bg[:, hitch_idx]

        col_cands = np.nonzero(max_r2_bg >= 0.85)[0]

        scenarios = [("null_model", 0.0, None, 0.0)]
        for pv in pve_list:
            scenarios.append(("true_coding", pv, g_code, code_max_r2))
            scenarios.append(("hitchhiking", pv, g_hitch, hitch_r2))
            if len(col_cands) > 0:
                ci = int(rng.choice(col_cands))
                scenarios.append(("hitch_collinear", pv, Z_bg[:, ci], float(max_r2_bg[ci])))

        for scen, pv, gsig, cr2 in scenarios:
            y = y0 + (np.sqrt(pv / (1.0 - pv)) * gsig if pv > 0 else 0.0)
            yt = W @ y
            D = dict(W=W32, Xt=Xt, yt=yt, Q0=Q0)
            rfh = scan(yt, Xt, W, lab_c[None, :], want_effects=False)[0]
            p_fh = rfh["P"] if rfh else 1.0
            lbc = bi_lbc_evaluate(Z_code, Z_bg, D, bg_var=a.bg_var, max_bg_pcs=a.max_bg_pcs, n_ctrl=0, rng=rng)
            p_single_ctrl_cond, _, _ = _kernel_score_test(Z_code, W32, Xt_ctrl1, yt)

            rows_out.append(
                dict(
                    gene_id=gid, n_code=Z_code.shape[1], n_cls=ncls_c,
                    causal_r2_bg=round(cr2, 3), scenario=scen, pve=pv,
                    P_fh_class=p_fh, P_code_kern=lbc["P_code"],
                    P_single_ctrl_cond=p_single_ctrl_cond,
                    P_lbc=lbc["P_lbc"], P_rev=lbc["P_rev"],
                    csi=lbc["csi"], status=lbc["status"],
                )
            )
        done += 1
        if done % 25 == 0:
            print(f"bench-lbc: {done}/{a.n_genes} genes done", flush=True)

    df = pd.DataFrame(rows_out)
    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(a.out, index=False)

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if len(df):
        def _agg(g):
            nn = len(g)
            k_fh = int((g.P_fh_class < 1e-3).sum())
            k_ck = int((g.P_code_kern < 1e-3).sum())
            k_sc = int(((g.P_code_kern < 1e-3) & (g.P_single_ctrl_cond < 0.01)).sum())
            k_lb = int(((g.P_code_kern < 1e-3) & (g.P_lbc < 0.01)).sum())
            k_st = int(((g.P_code_kern < 1e-3) & (g.P_lbc < 0.01) & (g.P_lbc < g.P_rev)).sum())
            lo, hi = _wilson_ci(k_lb, nn)
            return pd.Series({
                "n": nn,
                "FH_class": k_fh / nn,
                "CodeKern": k_ck / nn,
                "SingleCtrl": k_sc / nn,
                "BiLBC": k_lb / nn,
                "BiLBC_CI95": f"[{lo:.3f},{hi:.3f}]",
                "BiLBC_strict": k_st / nn,
                "median_CSI": g.csi.median(),
            })

        print(f"\n=== Bi-LBC Specificity Benchmark (elapsed {time.time()-t0:.1f}s, peak RAM {peak_bytes/(1024*1024):.1f} MB) ===")
        print(df.groupby(["scenario", "pve"]).apply(_agg).round(3).to_string())


if __name__ == "__main__":
    main()
