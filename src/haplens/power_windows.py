"""Multi-architecture power, calibration and runtime benchmark for window association tests."""

import argparse
import time
import tracemalloc
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import fdtrc

from haplens.kernel import acat_combine, dual_eigvalsh, pval_wchi2
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


def main(argv=None):
    t_wall = time.time()
    tracemalloc.start()
    ap = argparse.ArgumentParser(
        prog="haplens power-windows",
        description="Multi-architecture power and runtime benchmark of window tests.",
    )
    ap.add_argument("--no-sex-traits", default="", help="comma-separated traits without sex covariate")
    ap.add_argument("--chroms", default="1-38", help="chromosomes analysed")
    ap.add_argument("--bfile", required=True, help="PLINK binary genotype prefix")
    ap.add_argument("--windows", required=True, help="window table (.blocks.det)")
    ap.add_argument("--pheno", required=True, help="phenotype CSV")
    ap.add_argument("--rel", required=True, help="PLINK square GRM prefix")
    ap.add_argument("--trait", required=True, help="trait name")
    ap.add_argument("--n-win", type=int, default=200, help="number of windows to simulate")
    ap.add_argument(
        "--arch",
        default="1_causal,3_causal_same,3_causal_opp,2_epistasis",
        help="comma-separated architectures: 1_causal,3_causal_same,3_causal_opp,3_causal_het,2_epistasis",
    )
    ap.add_argument("--pve", default="0.005,0.01,0.02,0.04", help="comma-separated causal PVE values")
    ap.add_argument("--n-tests", type=int, default=18836, help="number of genome-wide windows for Bonferroni")
    ap.add_argument("--max-snp", type=int, default=0, help="max SNPs per window (0 = ALL SNPs)")
    ap.add_argument("--max-tags", type=int, default=4, help="max tag SNPs for haplotype-class test")
    ap.add_argument("--min-n", type=int, default=30, help="min individuals per haplotype class")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default="power.csv")
    a = ap.parse_args(argv)

    no_sex = set(filter(None, a.no_sex_traits.split(",")))
    archs = [x.strip() for x in a.arch.split(",") if x.strip()]
    pves = [float(x) for x in a.pve.split(",") if x.strip()]
    rng = np.random.default_rng(a.seed)

    fam = pd.read_csv(a.bfile + ".fam", sep=r"\s+", header=None, dtype=str)
    n_all = len(fam)
    bpl = (n_all + 3) // 4
    bim = pd.read_csv(
        a.bfile + ".bim", sep=r"\s+", header=None,
        names=["chr", "id", "cm", "pos", "A1", "A2"], dtype={"chr": str},
    )
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

    y0 = d[t].values.astype(float)
    y0 = (y0 - y0.mean()) / max(y0.std(), 1e-12)
    X = build_design_matrix(d, t, no_sex_traits=no_sex)
    mats, _ = build_random_matrices(d, R, rpos, random_spec="grm,dam")
    _, V = reml_fit(y0, X, mats)
    W = whiten(V)
    Xt = W @ X
    Xpinv = np.linalg.pinv(Xt)
    W32 = W.astype(np.float32)
    dof = n - int(np.linalg.matrix_rank(Xt))
    df_snp = max(dof - 1, 1)

    CH = chrom_list(a.chroms)
    Wd = pd.read_csv(a.windows, sep=r"\s+", dtype={"CHR": str})
    Wd = Wd[Wd.CHR.isin(CH)]
    pick = Wd.sample(n=min(a.n_win * 3, len(Wd)), random_state=a.seed)

    chr_slices = {}
    bim_chr = bim.chr.values
    bim_pos = bim.pos.values
    for c in CH:
        idx_c = np.nonzero(bim_chr == c)[0]
        if len(idx_c):
            ord_c = idx_c[np.argsort(bim_pos[idx_c], kind="stable")]
            chr_slices[c] = (ord_c, bim_pos[ord_c])

    timings = dict(kernel=0.0, skat_beta=0.0, cls=0.0, single_snp=0.0, acat_v=0.0)
    rows = []
    done = 0

    for w in pick.itertuples():
        if done >= a.n_win:
            break
        if w.CHR not in chr_slices:
            continue
        ord_c, pos_c = chr_slices[w.CHR]
        i0, i1 = np.searchsorted(pos_c, w.BP1), np.searchsorted(pos_c, w.BP2, side="right")
        idx = ord_c[i0:i1]
        if len(idx) < 10:
            continue
        raw = np.asarray(bed[idx])
        g = LUT[(raw[:, :, None] >> SH & 3).reshape(len(idx), -1)[:, :n_all][:, keep]][:, rowsel]
        ok = g >= 0
        af = np.maximum(g, 0).sum(1) / np.maximum(2 * ok.sum(1), 1)
        maf = np.minimum(af, 1.0 - af)
        good = np.nonzero((ok.mean(1) >= 0.9) & (maf >= 0.05))[0]
        if len(good) < 5:
            continue
        if a.max_snp > 0 and len(good) > a.max_snp:
            good = np.sort(rng.choice(good, a.max_snp, replace=False))
        G = g[good]
        maf_g = maf[good]
        m_snp = len(good)

        Xc = np.where(G >= 0, G, np.nan).astype(float)
        mu = np.nanmean(Xc, 1, keepdims=True)
        Xc = np.where(np.isnan(Xc), mu, Xc) - mu
        sd = Xc.std(1)
        sd[sd == 0] = 1.0
        Xs = Xc / sd[:, None]
        Z = Xs.T

        w_beta = stats.beta.pdf(maf_g, 1, 25)
        Z_beta = (Xc * w_beta[:, None]).T
        Z_beta = Z_beta / max(Z_beta.std(), 1e-12)

        kp = []
        for ii in np.argsort(-maf_g):
            if len(kp) >= a.max_tags:
                break
            if all(((Xs[ii] @ Xs[jj]) / n) ** 2 < 0.8 for jj in kp):
                kp.append(ii)
        Ttag = G[sorted(kp)]
        okc = (Ttag >= 0).all(0)
        if okc.sum() < 2 * a.min_n:
            continue
        _, inv, cnt = np.unique(Ttag[:, okc].T, axis=0, return_inverse=True, return_counts=True)
        kc = np.nonzero(cnt >= a.min_n)[0]
        if len(kc) < 2:
            continue
        lab = np.full(n, -1, np.int16)
        io = np.nonzero(okc)[0]
        for j, c_ in enumerate(kc[np.argsort(-cnt[kc])]):
            lab[io[inv == c_]] = j

        t0 = time.perf_counter()
        Zt = (W32 @ Z.astype(np.float32)).astype(np.float64)
        B = Zt - Xt @ (Xpinv @ Zt)
        lam_k = dual_eigvalsh(B)
        btb = np.maximum((B**2).sum(0), 1e-12)
        timings["kernel"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        Zt_b = (W32 @ Z_beta.astype(np.float32)).astype(np.float64)
        B_b = Zt_b - Xt @ (Xpinv @ Zt_b)
        lam_b = dual_eigvalsh(B_b)
        timings["skat_beta"] += time.perf_counter() - t0

        c1 = int(rng.choice(m_snp))
        g_1c = Xs[c1]

        cand_perm = rng.permutation(m_snp)
        c3 = []
        for ii in cand_perm:
            if all(((Xs[ii] @ Xs[jj]) / n) ** 2 < 0.3 for jj in c3):
                c3.append(int(ii))
            if len(c3) == 3:
                break
        if len(c3) < 3:
            c3 = list(rng.choice(m_snp, min(3, m_snp), replace=False))

        g_3same = Xs[c3].sum(0)
        g_3same = (g_3same - g_3same.mean()) / max(g_3same.std(), 1e-12)

        opp_signs = np.array([1.0, -1.0, 1.0])[: len(c3)]
        g_3opp = (Xs[c3] * opp_signs[:, None]).sum(0)
        g_3opp = (g_3opp - g_3opp.mean()) / max(g_3opp.std(), 1e-12)

        e1, e2 = c3[0], c3[min(1, len(c3) - 1)]
        inter = Xs[e1] * Xs[e2]
        E_main = np.column_stack([np.ones(n), Xs[e1], Xs[e2]])
        inter = inter - E_main @ np.linalg.lstsq(E_main, inter, rcond=None)[0]
        g_epi = (inter - inter.mean()) / max(inter.std(), 1e-12)

        arch_map = {
            "1_causal": g_1c,
            "3_causal_same": g_3same,
            "3_causal_opp": g_3opp,
            "3_causal_het": g_3opp,
            "2_epistasis": g_epi,
        }
        scenarios = [("null_model", 0.0, None)]
        for ar in archs:
            if ar in arch_map:
                for pv in pves:
                    scenarios.append((ar, pv, arch_map[ar]))

        for ar, pv, gsig in scenarios:
            y = y0 + (np.sqrt(pv / (1.0 - pv)) * gsig if pv > 0 else 0.0)
            yt = W @ y
            r0 = yt - Xt @ (Xpinv @ yt)
            rss0 = float(r0 @ r0)
            s2 = rss0 / max(dof, 1)

            t0 = time.perf_counter()
            btr = B.T @ r0
            pk = pval_wchi2(float((btr**2).sum()) / s2, lam_k)
            timings["kernel"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            pb = pval_wchi2(float(((B_b.T @ r0) ** 2).sum()) / s2, lam_b)
            timings["skat_beta"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            rc = scan(yt, Xt, W, lab[None, :], want_effects=False)[0]
            pc = rc["P"] if rc else 1.0
            timings["cls"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            rss1 = np.maximum(rss0 - (btr**2) / btb, 1e-12)
            f_s = ((rss0 - rss1) / rss1) * df_snp
            p_snps = fdtrc(1, df_snp, np.maximum(f_s, 0.0))
            p_min = float(p_snps.min())
            timings["single_snp"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            p_acatv = acat_combine(p_snps)
            timings["acat_v"] += time.perf_counter() - t0

            p_acat2 = acat_combine([pk, pc])
            p_omni = acat_combine([pk, p_acatv, pc])
            rows.append(
                dict(
                    trait=t, window=f"{w.CHR}:{w.BP1}", n_snp=m_snp, arch=ar, pve=pv,
                    P_kern=pk, P_skat_beta=pb, P_cls=pc, P_acat=p_acat2,
                    P_acatv=p_acatv, P_haplens_o=p_omni, P_min_snp=p_min,
                )
            )
        done += 1
        if done % 25 == 0:
            print(f"power-windows ({t}): {done}/{a.n_win} windows done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    thr = 0.05 / a.n_tests

    def _agg(g):
        nn = len(g)
        ko = int((g.P_haplens_o < thr).sum())
        lo, hi = _wilson_ci(ko, nn)
        return pd.Series({
            "n": nn,
            "HapLens_O": ko / nn,
            "HapLens_O_CI95": f"[{lo:.3f},{hi:.3f}]",
            "Kernel": (g.P_kern < thr).mean(),
            "ACAT_V": (g.P_acatv < thr).mean(),
            "SKAT_Beta": (g.P_skat_beta < thr).mean(),
            "Class": (g.P_cls < thr).mean(),
            "SNP_5e8": (g.P_min_snp < 5e-8).mean(),
        })

    print(f"trait {t}, n {n}, Bonferroni {thr:.2e}, elapsed {time.time()-t_wall:.1f}s, peak RAM {peak_bytes/(1024*1024):.1f} MB")
    print(res.groupby(["arch", "pve"]).apply(_agg).round(3).to_string())
    print("Component runtime (sec):", {k: round(v, 2) for k, v in timings.items()})


if __name__ == "__main__":
    main()
