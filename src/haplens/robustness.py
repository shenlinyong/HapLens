"""Leave-one-sire-out and sire split-half stability, haplotype groups and LD of selected windows."""

import argparse, os, numpy as np, pandas as pd
from scipy import linalg
from haplens.model import reml_fit
from haplens.kernel import pval_wchi2


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens robustness",
        description="Leave-one-sire-out and sire split-half stability, haplotype groups and LD of selected windows.",
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
    ap.add_argument("--ld-regions", default="")
    ap.add_argument("--splits", type=int, default=100, help="number of random sire split-halves")
    ap.add_argument("--maf", type=float, default=0.05, help="minimum minor allele frequency")
    ap.add_argument(
        "--max-snp",
        type=int,
        default=300,
        help="maximum number of SNPs per window (random subset above this)",
    )
    ap.add_argument("--seed", type=int, default=11, help="random seed")
    ap.add_argument("--out", default="loc", help="output directory or file prefix")
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

    def region_geno(c, s, e):
        idx = np.nonzero((bim.chr.values == c) & (bim.pos.values >= s) & (bim.pos.values <= e))[0]
        idx = idx[np.argsort(bim.pos.values[idx], kind="stable")]
        raw = np.asarray(bed[idx])
        g = LUT[(raw[:, :, None] >> SH & 3).reshape(len(idx), -1)[:, :n_all][:, keep]]
        ok = g >= 0
        af = np.maximum(g, 0).sum(1) / np.maximum(2 * ok.sum(1), 1)
        maf = np.minimum(af, 1 - af)
        good = (ok.mean(1) >= 0.9) & (maf >= a.maf)
        return (bim.pos.values[idx][good], g[good], maf[good])

    def standardize(g):
        X = np.where(g >= 0, g, np.nan).astype(float)
        mu = np.nanmean(X, 1, keepdims=True)
        X = np.where(np.isnan(X), mu, X) - mu
        sd = X.std(1)
        sd[sd == 0] = 1
        return (X / sd[:, None]).T

    def kernel_p(y, X, V, Z, idx):
        Vs = V[np.ix_(idx, idx)]
        L = linalg.cholesky(Vs, lower=True)
        W = linalg.solve_triangular(L, np.eye(len(idx)), lower=True)
        yt = W @ y[idx]
        Xt = W @ X[idx]
        Zt = W @ Z[idx]
        beta = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        r0 = yt - Xt @ beta
        s2 = float(r0 @ r0) / (len(idx) - Xt.shape[1])
        B = Zt - Xt @ (np.linalg.pinv(Xt) @ Zt)
        lam = np.linalg.eigvalsh(B.T @ B)
        return pval_wchi2(float(((Zt.T @ r0) ** 2).sum()) / s2, lam)

    wins = [w for w in a.windows.split(",") if w]
    WZ = {}
    for w in wins:
        c, r_ = w.split(":")
        s, e = map(int, r_.split("-"))
        pos, g, maf = region_geno(c, s, e)
        if len(pos) > a.max_snp:
            sel = np.sort(rng.choice(len(pos), a.max_snp, replace=False))
            pos, g = (pos[sel], g[sel])
        WZ[w] = (pos, g)
        print("window", w, "SNPs", len(pos), flush=True)
    rid = pd.read_csv(a.rel + ".rel.id", sep=r"\s+", header=None, dtype=str)[1].values
    R = np.loadtxt(a.rel + ".rel")
    rpos = {i: k for k, i in enumerate(rid)}
    rows, adj = ([], pd.DataFrame(index=iid))
    for t in a.traits.split(","):
        d = ph[ph[t].notna() & ph.sex.notna() & ph.index.isin(rid) & ph.index.isin(iid)]
        ids = d.index.values
        m_ = len(ids)
        rowsel = np.array([ipos[i] for i in ids])
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
        sr = d.sire.values
        comp, V = reml_fit(y, X, [K, (dm[:, None] == dm[None, :]).astype(float)])
        full = np.arange(m_)
        b0 = np.linalg.lstsq(X, y, rcond=None)[0]
        adj.loc[ids, t] = y - X @ b0
        sires = np.unique(sr)
        for w in wins:
            Z = standardize(WZ[w][1])[rowsel]
            p_full = kernel_p(y, X, V, Z, full)
            loso = np.array([kernel_p(y, X, V, Z, np.nonzero(sr != s_)[0]) for s_ in sires])
            worst = sires[np.argmax(loso)]
            pa, pb = ([], [])
            for _ in range(a.splits):
                perm = rng.permutation(sires)
                h1 = np.isin(sr, perm[: len(perm) // 2])
                pa.append(kernel_p(y, X, V, Z, np.nonzero(h1)[0]))
                pb.append(kernel_p(y, X, V, Z, np.nonzero(~h1)[0]))
            pa, pb = (np.array(pa), np.array(pb))
            rows.append(
                dict(
                    trait=t,
                    window=w,
                    n=m_,
                    P_full=p_full,
                    loso_min=loso.min(),
                    loso_max=loso.max(),
                    loso_worst_sire=worst,
                    half_median_P=np.median(np.r_[pa, pb]),
                    frac_half_P05=float((np.r_[pa, pb] < 0.05).mean()),
                    frac_both_P05=float(((pa < 0.05) & (pb < 0.05)).mean()),
                )
            )
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/robustness.csv", index=False)
    out = pd.DataFrame(index=iid)
    for w in wins:
        Z = standardize(WZ[w][1])
        U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        for k in range(3):
            out[f"{w}_PC{k + 1}"] = U[:, k] * S[k]
        pc = U[:, :2] * S[:2]
        cent = pc[rng.choice(len(pc), 3, replace=False)]
        for _ in range(50):
            lab = np.argmin(((pc[:, None, :] - cent[None]) ** 2).sum(2), 1)
            cent = np.array([pc[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(3)])
        order = np.argsort(-np.bincount(lab, minlength=3))
        remap = {int(o): i for i, o in enumerate(order)}
        out[f"{w}_cluster"] = [remap[int(l)] for l in lab]
    out = out.join(adj)
    out.index.name = "IID"
    out.to_csv(f"{a.out}/window_groups_adjpheno.csv")
    for reg in [r for r in a.ld_regions.split(",") if r]:
        c, r_ = reg.split(":")
        s, e = map(int, r_.split("-"))
        pos, g, maf = region_geno(c, s, e)
        sel = np.nonzero(maf >= 0.15)[0]
        if len(sel) > 150:
            sel = sel[np.round(np.linspace(0, len(sel) - 1, 150)).astype(int)]
        Z = standardize(g[sel])
        C = (Z.T @ Z / len(Z)) ** 2
        np.savez_compressed(f"{a.out}/ld_{c}_{s}_{e}.npz", pos=pos[sel], r2=C)
        print("LD region", reg, "SNPs", len(sel), flush=True)
    print("done")


if __name__ == "__main__":
    main()
