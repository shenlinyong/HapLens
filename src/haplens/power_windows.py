"""Power of the window tests with a simulated causal SNP."""

import argparse, numpy as np, pandas as pd
from haplens.model import reml_fit, whiten, scan
from haplens.kernel import pval_wchi2
from haplens.utils import chrom_list


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens power-windows", description="Power of the window tests with a simulated causal SNP."
    )
    ap.add_argument(
        "--no-sex-traits",
        default="",
        help="comma-separated traits analysed without the sex covariate (e.g. a trait measured at birth)",
    )
    ap.add_argument("--chroms", default="1-38", help="chromosomes analysed (default: autosomes 1-38)")
    ap.add_argument("--bfile", required=True, help="PLINK binary genotype prefix (.bed/.bim/.fam)")
    ap.add_argument(
        "--windows", required=True, help="window table (PLINK .blocks.det layout: CHR BP1 BP2 KB NSNPS SNPS)"
    )
    ap.add_argument(
        "--pheno",
        required=True,
        help="phenotype table with columns IID, sex, batch, sire, dam and one column per trait",
    )
    ap.add_argument("--rel", required=True, help="prefix of the PLINK square GRM (.rel and .rel.id)")
    ap.add_argument("--trait", required=True, help="trait name")
    ap.add_argument("--n-win", type=int, default=200, help="number of windows to simulate")
    ap.add_argument(
        "--pve",
        default="0.005,0.01,0.02,0.04",
        help="comma-separated causal effect sizes as fractions of the phenotypic variance",
    )
    ap.add_argument("--n-tests", type=int, default=18836)
    ap.add_argument(
        "--max-snp",
        type=int,
        default=300,
        help="maximum number of SNPs per window (random subset above this)",
    )
    ap.add_argument("--max-tags", type=int, default=4, help="maximum number of tag SNPs per window")
    ap.add_argument("--min-n", type=int, default=30, help="minimum number of individuals per haplotype class")
    ap.add_argument("--seed", type=int, default=21, help="random seed")
    ap.add_argument("--out", default="power.csv", help="output directory or file prefix")
    a = ap.parse_args(argv)
    no_sex = set(filter(None, a.no_sex_traits.split(",")))
    rng = np.random.default_rng(a.seed)
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
    t = a.trait
    rid = pd.read_csv(a.rel + ".rel.id", sep=r"\s+", header=None, dtype=str)[1].values
    R = np.loadtxt(a.rel + ".rel")
    rpos = {i: k for k, i in enumerate(rid)}
    d = ph[ph[t].notna() & ph.sex.notna() & ph.index.isin(rid) & ph.index.isin(fam[1].values)]
    ids = d.index.values
    n = len(ids)
    keep = fam[1].isin(ids).values
    order = pd.Series(np.arange(keep.sum()), index=fam[1].values[keep])
    rowsel = order.loc[ids].values
    bed = np.memmap(a.bfile + ".bed", dtype=np.uint8, mode="r", offset=3, shape=(len(bim), bpl))
    SH = np.array([0, 2, 4, 6], dtype=np.uint8)
    LUT = np.array([2, -1, 1, 0], dtype=np.int8)
    y0 = d[t].values.astype(float)
    y0 = (y0 - y0.mean()) / y0.std()
    cov = (
        [np.ones(n)]
        + ([d.sex.values.astype(float)] if t not in no_sex else [])
        + [(d.batch.values == b).astype(float) for b in sorted(d.batch.unique())[1:]]
    )
    X = np.column_stack(cov)
    K = R[np.ix_([rpos[i] for i in ids], [rpos[i] for i in ids])] + 0.001 * np.eye(n)
    dm = d.dam.values
    comp, V = reml_fit(y0, X, [K, (dm[:, None] == dm[None, :]).astype(float)])
    W = whiten(V)
    Xt = W @ X
    Xpinv = np.linalg.pinv(Xt)
    W32 = W.astype(np.float32)
    Wd = pd.read_csv(a.windows, sep=r"\s+", dtype={"CHR": str})
    Wd = Wd[Wd.CHR.isin(chrom_list(a.chroms))]
    pick = Wd.sample(n=min(a.n_win * 3, len(Wd)), random_state=a.seed)

    def acat(p1, p2):
        T = (
            np.tan((0.5 - np.clip(p1, 1e-300, 1 - 1e-16)) * np.pi)
            + np.tan((0.5 - np.clip(p2, 1e-300, 1 - 1e-16)) * np.pi)
        ) / 2
        return 1 / (np.pi * T) if T > 1000000000000000.0 else 0.5 - np.arctan(T) / np.pi

    def tests(y, Z, lab):
        yt = W @ y
        beta = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        r0 = yt - Xt @ beta
        s2 = float(r0 @ r0) / (n - Xt.shape[1])
        Zt = (W32 @ Z.astype(np.float32)).astype(np.float64)
        B = Zt - Xt @ (Xpinv @ Zt)
        lam = np.linalg.eigvalsh(B.T @ B)
        pk = pval_wchi2(float(((Zt.T @ r0) ** 2).sum()) / s2, lam)
        r = scan(yt, Xt, W, lab[None, :], want_effects=False)[0]
        pc = r["P"] if r else 1.0
        return (pk, pc, acat(pk, pc))

    rows = []
    done = 0
    for w in pick.itertuples():
        if done >= a.n_win:
            break
        idx = np.nonzero((bim.chr.values == w.CHR) & (bim.pos.values >= w.BP1) & (bim.pos.values <= w.BP2))[0]
        if len(idx) < 20:
            continue
        raw = np.asarray(bed[idx])
        g = LUT[(raw[:, :, None] >> SH & 3).reshape(len(idx), -1)[:, :n_all][:, keep]][:, rowsel]
        ok = g >= 0
        af = np.maximum(g, 0).sum(1) / np.maximum(2 * ok.sum(1), 1)
        maf = np.minimum(af, 1 - af)
        good = np.nonzero((ok.mean(1) >= 0.9) & (maf >= 0.05))[0]
        if len(good) < 5:
            continue
        if len(good) > a.max_snp:
            good = np.sort(rng.choice(good, a.max_snp, replace=False))
        G = g[good]
        Xs = np.where(G >= 0, G, np.nan).astype(float)
        mu = np.nanmean(Xs, 1, keepdims=True)
        Xs = np.where(np.isnan(Xs), mu, Xs) - mu
        sd = Xs.std(1)
        sd[sd == 0] = 1
        Z = (Xs / sd[:, None]).T
        C = Xs / sd[:, None] @ (Xs / sd[:, None]).T / n
        C = C**2
        kp = []
        for i in np.argsort(-np.minimum(af[good], 1 - af[good])):
            if len(kp) >= a.max_tags:
                break
            if all((C[i, j] < 0.8 for j in kp)):
                kp.append(i)
        T = G[sorted(kp)]
        okc = (T >= 0).all(0)
        lab = np.full(n, -1, np.int16)
        if okc.sum() < 2 * a.min_n:
            continue
        u, inv, cnt = np.unique(T[:, okc].T, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        kc = np.nonzero(cnt >= a.min_n)[0]
        if len(kc) < 2:
            continue
        io = np.nonzero(okc)[0]
        for j, c_ in enumerate(kc[np.argsort(-cnt[kc])]):
            lab[io[inv == c_]] = j
        cs = int(rng.choice(len(good)))
        dos = Xs[cs]
        dos = dos / dos.std()
        for pv in [float(x) for x in a.pve.split(",")] + [0.0]:
            y = y0 + (np.sqrt(pv / (1 - pv)) * dos if pv > 0 else 0.0)
            pk, pc, pa_ = tests(y, Z, lab)
            rows.append(
                dict(window=f"{w.CHR}:{w.BP1}", n_snp=len(good), pve=pv, P_kern=pk, P_cls=pc, P_acat=pa_)
            )
        done += 1
        if done % 20 == 0:
            print("windows done", done, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    thr = 0.05 / a.n_tests
    tab = res.groupby("pve").apply(
        lambda g: pd.Series(
            {
                "windows": len(g),
                **{
                    f"{nm}_bonf": (g[c] < thr).mean()
                    for nm, c in (("kernel", "P_kern"), ("class", "P_cls"), ("ACAT", "P_acat"))
                },
                **{
                    f"{nm}_p001": (g[c] < 0.001).mean()
                    for nm, c in (("kernel", "P_kern"), ("class", "P_cls"), ("ACAT", "P_acat"))
                },
            }
        )
    )
    print(f"trait {t}, n {n}, Bonferroni threshold {thr:.2e}")
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
