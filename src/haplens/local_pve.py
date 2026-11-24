"""Variance explained by a window (REML with a window kernel)."""

import argparse, numpy as np, pandas as pd
from haplens.model import reml_fit


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens local-pve", description="Variance explained by a window (REML with a window kernel)."
    )
    ap.add_argument(
        "--no-sex-traits",
        default="",
        help="comma-separated traits analysed without the sex covariate (e.g. a trait measured at birth)",
    )
    ap.add_argument("--bfile", required=True, help="PLINK binary genotype prefix (.bed/.bim/.fam)")
    ap.add_argument(
        "--pheno",
        required=True,
        help="phenotype table with columns IID, sex, batch, sire, dam and one column per trait",
    )
    ap.add_argument("--rel", required=True, help="prefix of the PLINK square GRM (.rel and .rel.id)")
    ap.add_argument("--traits", required=True, help="comma-separated trait names")
    ap.add_argument(
        "--windows", required=True, help="window table (PLINK .blocks.det layout: CHR BP1 BP2 KB NSNPS SNPS)"
    )
    ap.add_argument("--maf", type=float, default=0.05, help="minimum minor allele frequency")
    ap.add_argument(
        "--max-snp",
        type=int,
        default=0,
        help="maximum number of SNPs per window (random subset above this)",
    )
    ap.add_argument("--seed", type=int, default=3, help="random seed")
    ap.add_argument("--out", default="localh2.csv", help="output directory or file prefix")
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
    keep = fam[1].isin(ph.index).values
    iid = fam[1].values[keep]
    ipos = {i: k for k, i in enumerate(iid)}
    bed = np.memmap(a.bfile + ".bed", dtype=np.uint8, mode="r", offset=3, shape=(len(bim), bpl))
    SH = np.array([0, 2, 4, 6], dtype=np.uint8)
    LUT = np.array([2, -1, 1, 0], dtype=np.int8)
    Z = {}
    for w in a.windows.split(","):
        c, r_ = w.split(":")
        s, e = map(int, r_.split("-"))
        idx = np.nonzero((bim.chr.values == c) & (bim.pos.values >= s) & (bim.pos.values <= e))[0]
        g = LUT[(np.asarray(bed[idx])[:, :, None] >> SH & 3).reshape(len(idx), -1)[:, :n_all][:, keep]]
        ok = g >= 0
        f = np.maximum(g, 0).sum(1) / np.maximum(2 * ok.sum(1), 1)
        maf = np.minimum(f, 1 - f)
        good = np.nonzero((ok.mean(1) >= 0.9) & (maf >= a.maf))[0]
        if a.max_snp > 0 and len(good) > a.max_snp:
            good = np.sort(rng.choice(good, a.max_snp, replace=False))
        X = np.where(g[good] >= 0, g[good], np.nan).astype(float)
        mu = np.nanmean(X, 1, keepdims=True)
        X = np.where(np.isnan(X), mu, X) - mu
        sd = X.std(1)
        sd[sd == 0] = 1
        Z[w] = (X / sd[:, None]).T
    rid = pd.read_csv(a.rel + ".rel.id", sep=r"\s+", header=None, dtype=str)[1].values
    R = np.loadtxt(a.rel + ".rel")
    rpos = {i: k for k, i in enumerate(rid)}
    rows = []
    for t in a.traits.split(","):
        d = ph[ph[t].notna() & ph.sex.notna() & ph.index.isin(rid) & ph.index.isin(iid)]
        ids = d.index.values
        n = len(ids)
        rs = np.array([ipos[i] for i in ids])
        y = d[t].values.astype(float)
        y = (y - y.mean()) / y.std()
        X = np.column_stack(
            [np.ones(n)]
            + ([d.sex.values.astype(float)] if t not in no_sex else [])
            + [(d.batch.values == b).astype(float) for b in sorted(d.batch.unique())[1:]]
        )
        K = R[np.ix_([rpos[i] for i in ids], [rpos[i] for i in ids])] + 0.001 * np.eye(n)
        dm = d.dam.values
        D = (dm[:, None] == dm[None, :]).astype(float)
        for w, Zw in Z.items():
            Kw = Zw[rs] @ Zw[rs].T / Zw.shape[1]
            comp, V = reml_fit(y, X, [K, D, Kw])
            tot = comp.sum()
            rows.append(
                dict(
                    trait=t,
                    window=w,
                    n=n,
                    var_grm=comp[0],
                    var_dam=comp[1],
                    var_window=comp[2],
                    var_resid=comp[3],
                    local_PVE=comp[2] / tot,
                    grm_h2=comp[0] / tot,
                )
            )
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
