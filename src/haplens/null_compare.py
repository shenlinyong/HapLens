"""Compare random-effect structures of the null model by the inflation of the gene-level test."""

import argparse, numpy as np, pandas as pd
from scipy import stats
from haplens.model import reml_fit, whiten, scan


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens null-compare",
        description="Compare random-effect structures of the null model by the inflation of the gene-level test.",
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
    ap.add_argument(
        "--traits", required=True, help="comma-separated traits, including names defined by --composite"
    )
    ap.add_argument(
        "--composite",
        action="append",
        default=[],
        metavar="NAME=T1,T2,...",
        help="define a trait as the mean of the standardised listed traits (repeatable)",
    )
    ap.add_argument("--out", default="sens.csv", help="output directory or file prefix")
    a = ap.parse_args(argv)
    no_sex = set(filter(None, a.no_sex_traits.split(",")))
    z = np.load(a.fh, allow_pickle=True)
    genes = z["genes"]
    FH = z["fh"]
    iid_fh = z["iid"].astype(str)
    ph = pd.read_csv(a.pheno, dtype={"IID": str}).set_index("IID")
    for spec in a.composite:
        name, members = spec.split("=")
        cols = [c for c in members.split(",") if c in ph.columns]
        ph[name] = ((ph[cols] - ph[cols].mean()) / ph[cols].std()).mean(axis=1, skipna=True)
    rid = pd.read_csv(a.rel + ".rel.id", sep=r"\s+", header=None, dtype=str)[1].values
    R = np.loadtxt(a.rel + ".rel")
    rpos = {i: k for k, i in enumerate(rid)}
    fpos = {i: k for k, i in enumerate(iid_fh)}
    lam = lambda p: float(np.median(stats.chi2.isf(p, 1)) / 0.4549364)
    thr = 0.05 / len(genes)
    rows = []
    for t in a.traits.split(","):
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
        dm, sr = (d.dam.values, d.sire.values)
        D = (dm[:, None] == dm[None, :]).astype(float)
        S = (sr[:, None] == sr[None, :]).astype(float)
        L = FH[:, [fpos[i] for i in ids]]
        for name, mats in (("grm+dam", [K, D]), ("grm+dam+sire", [K, D, S]), ("dam+sire", [D, S])):
            comp, V = reml_fit(y, X, mats)
            W = whiten(V)
            res = scan(W @ y, W @ X, W, L, want_effects=False)
            P = np.array([x["P"] for x in res if x])
            gg = np.array([g for g, x in zip(genes, res) if x])
            o = np.argsort(P)
            rows.append(
                dict(
                    trait=t,
                    model=name,
                    n=n,
                    lambda_=lam(P),
                    n_p001=int((P < 0.001).sum()),
                    n_bonf=int((P < thr).sum()),
                    top_gene=gg[o[0]],
                    top_P=float(P[o[0]]),
                    var_comp=";".join((f"{c:.3f}" for c in comp)),
                )
            )
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
