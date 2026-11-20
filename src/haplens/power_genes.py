"""Power of the gene-level haplotype test by simulation."""

import argparse, numpy as np, pandas as pd
from haplens.model import reml_fit, whiten, scan


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens power-genes", description="Power of the gene-level haplotype test by simulation."
    )
    ap.add_argument(
        "--no-sex-traits",
        default="",
        help="comma-separated traits analysed without the sex covariate (e.g. a trait measured at birth)",
    )
    ap.add_argument(
        "--fh", required=True, help="haplotype file written by build-genes or build-windows (.npz)"
    )
    ap.add_argument(
        "--pheno",
        required=True,
        help="phenotype table with columns IID, sex, batch, sire, dam and one column per trait",
    )
    ap.add_argument("--rel", required=True, help="prefix of the PLINK square GRM (.rel and .rel.id)")
    ap.add_argument("--trait", required=True, help="trait name")
    ap.add_argument("--per-bin", type=int, default=60)
    ap.add_argument(
        "--pve",
        default="0.005,0.01,0.02,0.04",
        help="comma-separated causal effect sizes as fractions of the phenotypic variance",
    )
    ap.add_argument("--seed", type=int, default=3, help="random seed")
    ap.add_argument("--out", default="sim.csv", help="output directory or file prefix")
    a = ap.parse_args(argv)
    no_sex = set(filter(None, a.no_sex_traits.split(",")))
    rng = np.random.default_rng(a.seed)
    z = np.load(a.fh, allow_pickle=True)
    genes = z["genes"]
    FH = z["fh"]
    iid_fh = z["iid"].astype(str)
    ph = pd.read_csv(a.pheno, dtype={"IID": str}).set_index("IID")
    t = a.trait
    rid = pd.read_csv(a.rel + ".rel.id", sep=r"\s+", header=None, dtype=str)[1].values
    R = np.loadtxt(a.rel + ".rel")
    rpos = {i: k for k, i in enumerate(rid)}
    fpos = {i: k for k, i in enumerate(iid_fh)}
    d = ph[ph[t].notna() & ph.sex.notna() & ph.index.isin(rid) & ph.index.isin(iid_fh)]
    ids = d.index.values
    n = len(ids)
    y = d[t].values.astype(float)
    y = (y - y.mean()) / y.std()
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
    L = FH[:, [fpos[i] for i in ids]]
    thr = 0.05 / len(genes)

    def ncls(row):
        u, c = np.unique(row[row >= 0], return_counts=True)
        return (len(u), np.sort(c)[::-1])

    info = [ncls(L[g]) for g in range(len(genes))]
    bins = {
        "2 classes (df 1)": [g for g in range(len(genes)) if info[g][0] == 2],
        "3-6 classes": [g for g in range(len(genes)) if 3 <= info[g][0] <= 6],
        ">=7 classes": [g for g in range(len(genes)) if info[g][0] >= 7],
    }
    rows = []
    for name, pool in bins.items():
        pool = [g for g in pool if info[g][1][1] >= 20]
        pick = rng.choice(pool, size=min(a.per_bin, len(pool)), replace=False)
        for g in pick:
            lab = L[g]
            u, c = np.unique(lab[lab >= 0], return_counts=True)
            order = np.argsort(-c)
            cls = u[order[1]]
            f = c[order[1]] / (lab >= 0).sum()
            for pv in [float(x) for x in a.pve.split(",")] + [0.0]:
                beta = np.sqrt(pv / (f * (1 - f))) if pv > 0 else 0.0
                y2 = y + beta * (lab == cls)
                r = scan(W @ y2, Xt, W, lab[None, :])[0]
                if r is None:
                    continue
                est = [e for e in r["eff"] if e[0] == cls]
                rows.append(
                    dict(
                        bin=name,
                        gene=genes[g],
                        k=info[g][0],
                        freq=f,
                        pve=pv,
                        beta=beta,
                        P=r["P"],
                        est=est[0][1] if est else np.nan,
                        se=est[0][2] if est else np.nan,
                    )
                )
    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    res["hit_bonf"] = res.P < thr
    res["hit_1e3"] = res.P < 0.001
    print(f"trait {t}, n {n}, Bonferroni threshold {thr:.2e}")
    print(
        res.groupby(["bin", "pve"])
        .agg(
            genes=("P", "size"),
            power_bonf=("hit_bonf", "mean"),
            power_p001=("hit_1e3", "mean"),
            median_freq=("freq", "median"),
        )
        .round(3)
        .to_string()
    )
    nz = res[res.pve > 0]
    print(
        "effect estimate / true effect (median, among P<1e-3):",
        round(float((nz[nz.hit_1e3].est / nz[nz.hit_1e3].beta).median()), 2),
    )


if __name__ == "__main__":
    main()
