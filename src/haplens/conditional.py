"""Window kernel test conditional on lead SNPs."""

import argparse, os, numpy as np, pandas as pd
from scipy import linalg
from haplens.model import reml_fit
from haplens.kernel import dual_eigvalsh, pval_wchi2


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens conditional", description="Window kernel test conditional on lead SNPs."
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
        "--pairs", action="append", default=[], help="window=snp1+snp2 (chr:start-end=chr:pos[+chr:pos])"
    )
    ap.add_argument("--ld-snps", default="")
    ap.add_argument("--maf", type=float, default=0.05, help="minimum minor allele frequency")
    ap.add_argument(
        "--max-snp",
        type=int,
        default=0,
        help="maximum number of SNPs per window (random subset above this)",
    )
    ap.add_argument("--seed", type=int, default=3, help="random seed")
    ap.add_argument("--out", default="cond", help="output directory or file prefix")
    a = ap.parse_args(argv)
    no_sex = set(filter(None, a.no_sex_traits.split(",")))
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
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

    def geno(idx):
        return LUT[(np.asarray(bed[idx])[:, :, None] >> SH & 3).reshape(len(idx), -1)[:, :n_all][:, keep]]

    def snp(s):
        c, p = s.split(":")
        i = np.nonzero((bim.chr.values == c) & (bim.pos.values == int(p)))[0]
        if len(i) == 0:
            raise SystemExit(f"SNP {s} not found")
        g = geno(i[:1])[0].astype(float)
        g[g < 0] = np.nanmean(np.where(g >= 0, g, np.nan))
        return g

    def window_Z(w):
        c, r_ = w.split(":")
        s, e = map(int, r_.split("-"))
        idx = np.nonzero((bim.chr.values == c) & (bim.pos.values >= s) & (bim.pos.values <= e))[0]
        g = geno(idx)
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
        return (X / sd[:, None]).T

    def kernel_p(y, X, V, Z):
        L = linalg.cholesky(V, lower=True)
        W = linalg.solve_triangular(L, np.eye(len(y)), lower=True)
        yt = W @ y
        Xt = W @ X
        Zt = W @ Z
        beta = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        r0 = yt - Xt @ beta
        s2 = float(r0 @ r0) / (len(y) - np.linalg.matrix_rank(Xt))
        B = Zt - Xt @ (np.linalg.pinv(Xt) @ Zt)
        return pval_wchi2(float(((Zt.T @ r0) ** 2).sum()) / s2, dual_eigvalsh(B))

    pairs = [p.split("=") for p in a.pairs]
    WZ = {w: window_Z(w) for w, _ in pairs}
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
        comp, V = reml_fit(y, X, [K, (dm[:, None] == dm[None, :]).astype(float)])
        for w, lead in pairs:
            Z = WZ[w][rs]
            G = np.column_stack([snp(s)[rs] for s in lead.split("+")])
            p0 = kernel_p(y, X, V, Z)
            p1 = kernel_p(y, np.column_stack([X, G]), V, Z)
            rows.append(dict(trait=t, window=w, conditioned_on=lead, P_unconditional=p0, P_conditional=p1))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/conditional.csv", index=False)
    if a.ld_snps:
        ss = a.ld_snps.split(",")
        G = np.column_stack([snp(s) for s in ss])
        C = np.corrcoef(G.T) ** 2
        pd.DataFrame(C, index=ss, columns=ss).to_csv(f"{a.out}/ld_r2.csv")
        print(pd.DataFrame(C, index=ss, columns=ss).round(2))


if __name__ == "__main__":
    main()
