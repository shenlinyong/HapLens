"""Relate GRM principal components to the pedigree and design factors."""

import argparse, os, numpy as np, pandas as pd
from scipy import stats
from scipy.cluster.vq import kmeans2


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens pc-check",
        description="Relate GRM principal components to the pedigree and design factors.",
    )
    ap.add_argument("--pcs", required=True)
    ap.add_argument(
        "--pheno",
        required=True,
        help="phenotype table with columns IID, sex, batch, sire, dam and one column per trait",
    )
    ap.add_argument("--out", default="pc_check", help="output directory or file prefix")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pc = pd.read_csv(a.pcs, dtype={"IID": str}).set_index("IID")
    ph = pd.read_csv(a.pheno, dtype={"IID": str}).set_index("IID")
    d = pc.join(ph[["sex", "batch", "sire", "dam"]], how="inner")
    print("n =", len(d))
    rows = []
    for k in range(1, 6):
        for f in ("sire", "dam", "batch", "sex"):
            g = d[[f"PC{k}", f]].dropna()
            grp = [x.values for _, x in g.groupby(f)[f"PC{k}"] if len(x) > 1]
            tot = ((g[f"PC{k}"] - g[f"PC{k}"].mean()) ** 2).sum()
            within = sum((((x - x.mean()) ** 2).sum() for x in grp))
            r2 = 1 - within / tot
            dfb = len(grp) - 1
            dfw = len(g) - len(grp)
            adj = 1 - (1 - r2) * (len(g) - 1) / dfw
            rows.append(
                dict(
                    PC=k,
                    factor=f,
                    levels=len(grp),
                    R2=r2,
                    adjR2=adj,
                    P=stats.f.sf(r2 / dfb / ((1 - r2) / dfw), dfb, dfw),
                )
            )
    r = pd.DataFrame(rows)
    r.to_csv(f"{a.out}/pc_vs_design.csv", index=False)
    print(r.pivot(index="PC", columns="factor", values="adjR2").round(3).to_string())
    np.random.seed(1)
    cen, lab = kmeans2(d[["PC1", "PC2"]].values, 3, minit="++", seed=1)
    d["cluster"] = lab
    for f in ("sire", "batch", "dam"):
        ct = pd.crosstab(d[f], d.cluster)
        ct.to_csv(f"{a.out}/cluster_by_{f}.csv")
        pure = ct.max(1).sum() / ct.values.sum()
        print(
            f"\ncluster x {f}: share of individuals whose {f} sits in its modal cluster = {pure:.2f}; {f} levels spanning >1 cluster: {(ct.gt(0).sum(1) > 1).sum()} of {len(ct)}"
        )
    print("\ncluster sizes:", d.cluster.value_counts().to_dict())
    d[["PC1", "PC2", "cluster", "sire", "dam", "batch"]].to_csv(f"{a.out}/pcs_with_design.csv")


if __name__ == "__main__":
    main()
