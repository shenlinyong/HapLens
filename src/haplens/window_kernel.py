"""Regional relatedness-kernel score test of windows."""

import argparse, os, numpy as np, pandas as pd
from scipy import stats
from haplens.model import reml_fit, whiten
from haplens.kernel import pval_wchi2
from haplens.utils import chrom_list


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens test-windows", description="Regional relatedness-kernel score test of windows."
    )
    ap.add_argument(
        "--no-sex-traits",
        default="",
        help="comma-separated traits analysed without the sex covariate (e.g. a trait measured at birth)",
    )
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
    ap.add_argument("--traits", default=None, help="comma-separated trait names")
    ap.add_argument("--chroms", default="1-38", help="chromosomes analysed, e.g. 1-38 or 1-5,Z")
    ap.add_argument("--maf", type=float, default=0.05, help="minimum minor allele frequency")
    ap.add_argument("--call", type=float, default=0.9, help="minimum call rate")
    ap.add_argument(
        "--max-snp",
        type=int,
        default=300,
        help="maximum number of SNPs per window (random subset above this)",
    )
    ap.add_argument("--min-snp", type=int, default=3, help="minimum number of SNPs per window")
    ap.add_argument("--perm", type=int, default=10, help="number of sire-stratified permutations")
    ap.add_argument("--seed", type=int, default=5, help="random seed")
    ap.add_argument("--out", default="kern", help="output directory or file prefix")
    a = ap.parse_args(argv)
    no_sex = set(filter(None, a.no_sex_traits.split(",")))
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
    n = len(iid)
    ipos = {i: k for k, i in enumerate(iid)}
    bed = np.memmap(a.bfile + ".bed", dtype=np.uint8, mode="r", offset=3, shape=(len(bim), bpl))
    SH = np.array([0, 2, 4, 6], dtype=np.uint8)
    LUT = np.array([2, -1, 1, 0], dtype=np.int8)

    def dosage(rows):
        raw = np.asarray(bed[rows])
        return LUT[(raw[:, :, None] >> SH & 3).reshape(len(rows), -1)[:, :n_all][:, keep]]

    rid = pd.read_csv(a.rel + ".rel.id", sep=r"\s+", header=None, dtype=str)[1].values
    R = np.loadtxt(a.rel + ".rel")
    rpos = {i: k for k, i in enumerate(rid)}
    traits = (
        a.traits.split(",")
        if a.traits
        else [c for c in ph.columns if c not in ("sex", "batch", "sire", "dam")]
    )
    T = {}
    for t in traits:
        d = ph[ph[t].notna() & ph.sex.notna() & ph.index.isin(rid) & ph.index.isin(iid)]
        ids = d.index.values
        m_ = len(ids)
        y = d[t].values.astype(float)
        y = (y - y.mean()) / y.std()
        cov = (
            [np.ones(m_)]
            + ([d.sex.values.astype(float)] if t not in no_sex else [])
            + [(d.batch.values == b).astype(float) for b in sorted(d.batch.unique())[1:]]
        )
        X = np.column_stack(cov)
        K = R[np.ix_([rpos[i] for i in ids], [rpos[i] for i in ids])] + 0.001 * np.eye(m_)
        dm = d.dam.values
        comp, V = reml_fit(y, X, [K, (dm[:, None] == dm[None, :]).astype(float)])
        W = whiten(V)
        yt = W @ y
        Xt = W @ X
        beta = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        r0 = yt - Xt @ beta
        s2 = float(r0 @ r0) / (m_ - Xt.shape[1])
        strata = pd.Series(np.arange(m_)).groupby(d.sire.values).apply(list)
        RP = np.empty((m_, a.perm))
        for k in range(a.perm):
            perm = np.arange(m_)
            for idxs in strata:
                idxs = np.array(idxs)
                perm[idxs] = rng.permutation(idxs)
            RP[:, k] = r0[perm]
        T[t] = dict(
            rows=np.array([ipos[i] for i in ids]),
            W=W.astype(np.float32),
            Xt=Xt,
            Xpinv=np.linalg.pinv(Xt),
            r0=r0,
            s2=s2,
            RP=RP,
        )
        print("trait", t, "n", m_, "var comp", np.round(comp, 3), flush=True)
    Wd = pd.read_csv(a.windows, sep=r"\s+", dtype={"CHR": str})
    out = {t: [] for t in traits}
    permP = {t: [] for t in traits}
    for c in CH:
        ridx = np.nonzero(bim.chr.values == c)[0]
        if len(ridx) == 0:
            continue
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
            if len(idx) > a.max_snp:
                idx = np.sort(rng.choice(idx, a.max_snp, replace=False))
            X = np.where(G[idx] >= 0, G[idx], np.nan).astype(float)
            mu = np.nanmean(X, 1, keepdims=True)
            X = np.where(np.isnan(X), mu, X) - mu
            sd = X.std(1)
            sd[sd == 0] = 1
            Z = (X / sd[:, None]).T
            for t, D in T.items():
                Zr = Z[D["rows"]]
                Zt = (D["W"] @ Zr.astype(np.float32)).astype(np.float64)
                B = Zt - D["Xt"] @ (D["Xpinv"] @ Zt)
                lam = np.linalg.eigvalsh(B.T @ B)
                q = float(((Zt.T @ D["r0"]) ** 2).sum()) / D["s2"]
                qp = ((Zt.T @ D["RP"]) ** 2).sum(0) / D["s2"]
                out[t].append((c, w.BP1, w.BP2, len(idx), q, pval_wchi2(q, lam)))
                permP[t] += [pval_wchi2(x, lam) for x in qp]
        print("chr", c, "windows so far", len(out[traits[0]]), flush=True)
    tag = a.chroms.replace(",", "_")
    summ = []
    for t in traits:
        r = pd.DataFrame(out[t], columns=["chr", "start", "end", "n_snp", "Q", "P"])
        r.to_csv(f"{a.out}/{t}_{tag}.csv", index=False)
        pp = np.array(permP[t])
        lam_ = lambda p: float(np.median(stats.chi2.isf(np.clip(p, 1e-300, 1), 1)) / 0.4549364)
        summ.append(
            dict(
                trait=t,
                chroms=a.chroms,
                windows=len(r),
                lam_obs=lam_(r.P.values),
                n_p001=int((r.P < 0.001).sum()),
                n_bonf_win=int((r.P < 0.05 / max(len(r), 1)).sum()),
                perm_lambda=lam_(pp),
                perm_p05=float((pp < 0.05).mean()),
                perm_p001=float((pp < 0.001).mean()),
            )
        )
        print(summ[-1], flush=True)
    pd.DataFrame(summ).to_csv(f"{a.out}/summary_{'-'.join(traits)[:60]}_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
