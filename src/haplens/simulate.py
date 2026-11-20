"""Simulate a small structured population for examples and tests.

Sires and dams carry two haplotypes per gene region; offspring receive one haplotype from each
parent. Every gene has a few functional SNPs (annotated as missense) and a block of neutral SNPs.
Trait T0 has no gene effects; trait T1 carries the effect of one haplotype class in a subset of genes.
The output follows the input formats of the other commands (PLINK bed/bim/fam, phenotype table, GRM,
annotation tables, window file).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def _pool(rng, k, n_hap=4):
    patterns = []
    while len(patterns) < n_hap:
        p = rng.integers(0, 2, k)
        if p.sum() > 0 and not any((p == q).all() for q in patterns):
            patterns.append(p)
    freq = rng.dirichlet(np.ones(n_hap) * 1.2)
    freq = np.maximum(freq, 0.03)
    return np.array(patterns), freq / freq.sum()


def _draw(rng, pool, size):
    patterns, freq = pool
    return patterns[rng.choice(len(freq), size=size, p=freq)]


def _pack(G):
    """Pack a (snps x individuals) 0/1/2 matrix into PLINK bed bytes (A1 = counted allele)."""
    code = np.where(G == 2, 0, np.where(G == 1, 2, 3)).astype(np.uint8)
    m, n = G.shape
    pad = (-n) % 4
    code = np.hstack([code, np.full((m, pad), 3, np.uint8)]).reshape(m, -1, 4)
    return (code[:, :, 0] | (code[:, :, 1] << 2) | (code[:, :, 2] << 4) | (code[:, :, 3] << 6)).astype(
        np.uint8
    )


def simulate(out, n_genes=40, n_func=3, n_neutral=30, n_sires=10, n_dams=60, per_dam=8, n_causal=8, seed=11):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    n = n_dams * per_dam
    func_pool = [_pool(rng, n_func) for _ in range(n_genes)]
    neut_pool = [_pool(rng, n_neutral, 5) for _ in range(n_genes)]
    sire_of_dam = np.repeat(np.arange(n_sires), n_dams // n_sires + 1)[:n_dams]

    def founders(m):
        return [
            (
                np.stack([_draw(rng, func_pool[g], m), _draw(rng, func_pool[g], m)], 1),
                np.stack([_draw(rng, neut_pool[g], m), _draw(rng, neut_pool[g], m)], 1),
            )
            for g in range(n_genes)
        ]

    sire_h, dam_h = founders(n_sires), founders(n_dams)
    dam = np.repeat(np.arange(n_dams), per_dam)
    sire = sire_of_dam[dam]
    Gf, Gn = [], []
    for g in range(n_genes):
        f = sire_h[g][0][sire, rng.integers(0, 2, n)] + dam_h[g][0][dam, rng.integers(0, 2, n)]
        z = sire_h[g][1][sire, rng.integers(0, 2, n)] + dam_h[g][1][dam, rng.integers(0, 2, n)]
        Gf.append(f.T)
        Gn.append(z.T)
    Gf, Gn = np.vstack(Gf), np.vstack(Gn)

    sex = rng.integers(0, 2, n)
    batch = rng.integers(1, 4, n)
    poly = Gn[rng.choice(len(Gn), min(400, len(Gn)), replace=False)].T.astype(float)
    poly = poly @ rng.normal(0, 0.05, poly.shape[1])
    y0 = 0.5 * sex + 0.2 * batch + rng.normal(0, 0.5, n_dams)[dam] + 0.6 * poly + rng.normal(0, 1, n)
    y1 = y0.copy()
    truth = {}
    for i, g in enumerate(rng.choice(n_genes, n_causal, replace=False)):
        cls = np.unique(Gf[g * n_func : (g + 1) * n_func].T, axis=0, return_inverse=True)[1].ravel()
        target = np.argsort(-np.bincount(cls))[1]
        effect = [0.5, 0.7, 0.9][i % 3]
        y1 += effect * (cls == target)
        truth[f"G{int(g)}"] = effect
    iid = [f"I{i:05d}" for i in range(n)]
    pd.DataFrame(dict(IID=iid, sex=sex, batch=batch, sire=sire, dam=dam, T0=y0, T1=y1)).to_csv(
        f"{out}/pheno.csv", index=False
    )

    bim, ann = [], []
    for g in range(n_genes):
        for j in range(n_func):
            p = (g + 1) * 100000 + j * 50
            bim.append(("1", f"1:{p}", 0, p, "T", "A"))
            ann.append(
                (
                    "1",
                    p,
                    f"1:{p}",
                    g * n_func + j,
                    "A",
                    "T",
                    f"G{g}",
                    f"GENE{g}",
                    "missense",
                    "A",
                    "X1Y",
                    True,
                )
            )
    for g in range(n_genes):
        for j in range(n_neutral):
            p = (g + 1) * 100000 + 20000 + j * 500
            bim.append(("1", f"1:{p}", 0, p, "T", "A"))
    pd.DataFrame(bim).to_csv(f"{out}/ex.bim", sep="\t", header=False, index=False)
    with open(f"{out}/ex.fam", "w") as fh:
        fh.write("".join(f"0\t{i}\t0\t0\t0\t-9\n" for i in iid))
    with open(f"{out}/ex.bed", "wb") as fh:
        fh.write(bytes([0x6C, 0x1B, 0x01]))
        fh.write(_pack(np.vstack([Gf, Gn])).tobytes())
    cols = [
        "chr",
        "pos",
        "id",
        "row",
        "ref",
        "alt",
        "gene_id",
        "gene_name",
        "klass",
        "tier",
        "aa",
        "alt_is_A1",
    ]
    pd.DataFrame(ann, columns=cols).to_csv(f"{out}/ann_variants.tsv", sep="\t", index=False)
    genes = [
        (f"G{g}", f"GENE{g}", "1", (g + 1) * 100000 - 1000, (g + 1) * 100000 + 3000, "+", 1)
        for g in range(n_genes)
    ]
    pd.DataFrame(genes, columns=["gene_id", "gene_name", "chr", "start", "end", "strand", "n_tx"]).to_csv(
        f"{out}/ann_genes.tsv", sep="\t", index=False
    )
    wins = "".join(
        f"1 {(g + 1) * 100000 - 1000} {(g + 1) * 100000 + 38000} 39.0 33 X\n" for g in range(n_genes)
    )
    with open(f"{out}/windows.det", "w") as fh:
        fh.write(" CHR  BP1  BP2  KB NSNPS SNPS\n" + wins)
    Z = Gn.astype(float)
    p = Z.mean(1) / 2
    Zs = (Z - 2 * p[:, None]) / np.sqrt(2 * p * (1 - p) + 1e-9)[:, None]
    np.savetxt(f"{out}/grm.rel", Zs.T @ Zs / len(Zs), fmt="%.5f")
    with open(f"{out}/grm.rel.id", "w") as fh:
        fh.write("".join(f"0\t{i}\n" for i in iid))
    json.dump(truth, open(f"{out}/truth.json", "w"), indent=1)
    return truth


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens simulate", description="Simulate a small structured population."
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-genes", type=int, default=40)
    ap.add_argument("--n-sires", type=int, default=10)
    ap.add_argument("--n-dams", type=int, default=60)
    ap.add_argument("--per-dam", type=int, default=8)
    ap.add_argument("--n-causal", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    truth = simulate(a.out, a.n_genes, 3, 30, a.n_sires, a.n_dams, a.per_dam, a.n_causal, a.seed)
    print(f"wrote {a.out}: {a.n_dams * a.per_dam} individuals, {a.n_genes} genes, {len(truth)} causal genes")


if __name__ == "__main__":
    main()
