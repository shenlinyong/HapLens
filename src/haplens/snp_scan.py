"""Single-SNP scan under the null mixed model."""

import argparse, os, numpy as np, pandas as pd
from scipy import stats
from haplens.model import reml_fit, whiten
from haplens.utils import chrom_list


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens snp-scan", description="Single-SNP scan under the null mixed model."
    )
    ap.add_argument(
        "--no-sex-traits",
        default="",
        help="comma-separated traits analysed without the sex covariate (e.g. a trait measured at birth)",
    )
    ap.add_argument("--chroms", default="1-38", help="chromosomes analysed (default: autosomes 1-38)")
    ap.add_argument("--bfile", required=True, help="PLINK binary genotype prefix (.bed/.bim/.fam)")
    ap.add_argument(
        "--pheno",
        required=True,
        help="phenotype table with columns IID, sex, batch, sire, dam and one column per trait",
    )
    ap.add_argument("--rel", required=True, help="prefix of the PLINK square GRM (.rel and .rel.id)")
    ap.add_argument("--traits", required=True, help="comma-separated trait names")
    ap.add_argument("--regions", default="", help="comma-separated regions chr:start-end")
    ap.add_argument("--maf", type=float, default=0.01, help="minimum minor allele frequency")
    ap.add_argument("--call", type=float, default=0.9, help="minimum call rate")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--top-p", type=float, default=1e-05)
    ap.add_argument("--out", default="snp", help="output directory or file prefix")
    a = ap.parse_args(argv)
    no_sex = set(filter(None, a.no_sex_traits.split(",")))
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
    auto = bim.chr.isin(chrom_list(a.chroms)).values
    rows = np.nonzero(auto)[0]
    M = len(rows)
    print("SNPs on chr1-38:", M, flush=True)
    ph = pd.read_csv(a.pheno, dtype={"IID": str}).set_index("IID")
    keep = fam[1].isin(ph.index).values
    iid = fam[1].values[keep]
    ipos = {i: k for k, i in enumerate(iid)}
    bed = np.memmap(a.bfile + ".bed", dtype=np.uint8, mode="r", offset=3, shape=(len(bim), bpl))
    SH = np.array([0, 2, 4, 6], dtype=np.uint8)
    LUT = np.array([2, -1, 1, 0], dtype=np.int8)
    rid = pd.read_csv(a.rel + ".rel.id", sep=r"\s+", header=None, dtype=str)[1].values
    R = np.loadtxt(a.rel + ".rel")
    rpos = {i: k for k, i in enumerate(rid)}
    T = {}
    for t in a.traits.split(","):
        d = ph[ph[t].notna() & ph.sex.notna() & ph.index.isin(rid) & ph.index.isin(iid)]
        ids = d.index.values
        n = len(ids)
        y = d[t].values.astype(float)
        sd_y = y.std()
        y = (y - y.mean()) / sd_y
        cov = (
            [np.ones(n)]
            + ([d.sex.values.astype(float)] if t not in no_sex else [])
            + [(d.batch.values == b).astype(float) for b in sorted(d.batch.unique())[1:]]
        )
        X = np.column_stack(cov)
        K = R[np.ix_([rpos[i] for i in ids], [rpos[i] for i in ids])] + 0.001 * np.eye(n)
        dm = d.dam.values
        comp, V = reml_fit(y, X, [K, (dm[:, None] == dm[None, :]).astype(float)])
        W = whiten(V)
        Xt = W @ X
        yt = W @ y
        Q, _ = np.linalg.qr(Xt)
        ry = yt - Q @ (Q.T @ yt)
        T[t] = dict(
            rows=np.array([ipos[i] for i in ids]),
            W=W.astype(np.float32),
            Q=Q.astype(np.float32),
            ry=ry.astype(np.float32),
            n=n,
            p=Xt.shape[1],
            rss0=float(ry @ ry),
            beta=np.full(M, np.nan, np.float32),
            se=np.full(M, np.nan, np.float32),
            P=np.full(M, np.nan, np.float32),
            maf=np.full(M, np.nan, np.float32),
            f1=np.full(M, np.nan, np.float32),
        )
        print("trait", t, "n", n, "variance components", np.round(comp, 3), flush=True)
    for s0 in range(0, M, a.chunk):
        idx = rows[s0 : s0 + a.chunk]
        raw = np.asarray(bed[idx])
        g = LUT[(raw[:, :, None] >> SH & 3).reshape(len(idx), -1)[:, :n_all][:, keep]]
        ok = g >= 0
        call = ok.mean(1)
        f1 = np.maximum(g, 0).sum(1, dtype=np.int64) / np.maximum(2 * ok.sum(1), 1)
        maf = np.minimum(f1, 1 - f1)
        good = (call >= a.call) & (maf >= a.maf)
        G = np.where(ok, g, np.round(2 * f1)[:, None]).astype(np.float32)
        for t, D in T.items():
            Gt = D["W"] @ G[good][:, D["rows"]].T
            Gr = Gt - D["Q"] @ (D["Q"].T @ Gt)
            gg = (Gr * Gr).sum(0)
            gy = Gr.T @ D["ry"]
            beta = gy / np.maximum(gg, 1e-12)
            df = D["n"] - D["p"] - 1
            rss1 = D["rss0"] - beta**2 * gg
            se = np.sqrt(np.maximum(rss1, 1e-12) / df / np.maximum(gg, 1e-12))
            P = 2 * stats.t.sf(np.abs(beta / se), df)
            sl = np.nonzero(good)[0] + s0
            D["beta"][sl] = beta
            D["se"][sl] = se
            D["P"][sl] = P
            D["maf"][sl] = maf[good]
            D["f1"][sl] = f1[good]
        if s0 // a.chunk % 50 == 0:
            print("chunk", s0 // a.chunk, "of", M // a.chunk + 1, flush=True)
    info = bim.iloc[rows][["chr", "pos", "id", "A1", "A2"]].reset_index(drop=True)
    regs = [r for r in a.regions.split(",") if r]
    for t, D in T.items():
        tab = info.assign(f1=D["f1"], maf=D["maf"], beta=D["beta"], se=D["se"], P=D["P"])
        ok = tab.P.notna()
        chi = stats.chi2.isf(tab.P[ok].values.astype(float), 1)
        lam = float(np.median(chi) / 0.4549364)
        top = tab[ok & (tab.P < a.top_p)]
        top.to_csv(f"{a.out}/{t}_top.csv.gz", index=False)
        for r in regs:
            c, rg = r.split(":")
            s, e = map(int, rg.split("-"))
            tab[ok & (tab.chr == c) & (tab.pos >= s) & (tab.pos <= e)].to_csv(
                f"{a.out}/{t}_region_{c}_{s}_{e}.csv.gz", index=False
            )
        np.savez_compressed(f"{a.out}/{t}_P.npz", P=D["P"], beta=D["beta"], se=D["se"])
        print(
            f"{t}: tested {int(ok.sum())}, lambda {lam:.3f}, P<1e-5: {int((tab.P < 1e-05).sum())}, P<5e-8: {int((tab.P < 5e-08).sum())}",
            flush=True,
        )


if __name__ == "__main__":
    main()
