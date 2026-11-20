"""Export summary tables: minimum SNP P per window, GRM principal components, gene-level results of regions."""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from haplens.utils import chrom_list


def window_min_p(a):
    bim = pd.read_csv(
        a.bfile + ".bim",
        sep=r"\s+",
        header=None,
        names=["chr", "id", "cm", "pos", "A1", "A2"],
        dtype={"chr": str},
    )
    chroms = chrom_list(a.chroms)
    info = bim.loc[bim.chr.isin(chroms).values, ["chr", "pos"]].reset_index(drop=True)
    info["w"] = (info.pos - 1) // a.window_size
    W = pd.read_csv(a.windows, sep=r"\s+", dtype={"CHR": str})
    W = W[W.CHR.isin(chroms)].copy()
    W["w"] = (W.BP1 - 1) // a.window_size
    out = (
        W[["CHR", "BP1", "BP2"]]
        .rename(columns={"CHR": "chr", "BP1": "start", "BP2": "end"})
        .reset_index(drop=True)
    )
    key = W.CHR.astype(str) + ":" + W.w.astype(str)
    for f in sorted(glob.glob(f"{a.snpdir}/*_P.npz")):
        trait = os.path.basename(f).split("_")[0]
        d = info.assign(P=np.load(f)["P"]).dropna(subset=["P"])
        g = d.groupby([d.chr, d.w]).P.agg(["min", "size"])
        g.index = [f"{c}:{w}" for c, w in g.index]
        out[f"minP_{trait}"] = key.map(g["min"]).values
        out["n_snp"] = key.map(g["size"]).values
    out.to_csv(f"{a.out}/window_minP.csv", index=False)


def grm_pcs(a, n_pc=10):
    ids = pd.read_csv(a.rel + ".rel.id", sep=r"\s+", header=None, dtype=str)[1].values
    w, v = np.linalg.eigh(np.loadtxt(a.rel + ".rel"))
    o = np.argsort(-w)[:n_pc]
    pcs = pd.DataFrame(v[:, o] * np.sqrt(np.maximum(w[o], 0)), columns=[f"PC{i + 1}" for i in range(n_pc)])
    pcs.insert(0, "IID", ids)
    pcs.to_csv(f"{a.out}/grm_pcs.csv", index=False)
    pd.DataFrame({"PC": np.arange(1, n_pc + 1), "variance_explained": w[o] / w.clip(0).sum()}).to_csv(
        f"{a.out}/grm_pc_variance.csv", index=False
    )


def region_genes(a):
    G = pd.read_csv(a.ann_genes, sep="\t", dtype={"chr": str}).set_index("gene_id")
    rows = []
    for spec in a.tier:
        tier, rest = spec.split("=")
        genes_tsv, res_dir = rest.split(":")
        gm = pd.read_csv(genes_tsv, sep="\t", dtype={"chr": str}).set_index("gene_id")
        for reg in a.regions.split(","):
            c, rng_ = reg.split(":")
            s, e = map(int, rng_.split("-"))
            inside = G[(G.chr == c) & (G.end >= s) & (G.start <= e)].index
            for f in glob.glob(f"{res_dir}/[A-Z]*.csv"):
                trait = os.path.basename(f)[:-4]
                d = pd.read_csv(f).set_index("gene_id")
                for g in inside:
                    if g in d.index and g in gm.index:
                        rows.append(
                            dict(
                                tier=tier,
                                gene_id=g,
                                gene=G.loc[g, "gene_name"],
                                chr=c,
                                start=G.loc[g, "start"],
                                end=G.loc[g, "end"],
                                trait=trait,
                                P=d.loc[g, "P"],
                                df=d.loc[g, "df"],
                                pve=d.loc[g, "pve"],
                                n_var=gm.loc[g, "n_var"],
                                n_fh=gm.loc[g, "n_fh"],
                            )
                        )
    pd.DataFrame(rows).to_csv(f"{a.out}/region_gene_tests.csv", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens export-tables",
        description="Export the minimum SNP P per window, GRM principal components and gene-level results of regions.",
    )
    ap.add_argument("--rel", required=True, help="prefix of the PLINK square GRM (.rel and .rel.id)")
    ap.add_argument("--out", default="tables", help="output directory")
    ap.add_argument("--bfile", help="PLINK prefix; with --windows and --snpdir writes window_minP.csv")
    ap.add_argument("--windows", help="window table (PLINK .blocks.det layout)")
    ap.add_argument("--snpdir", help="directory written by snp-scan")
    ap.add_argument("--window-size", type=int, default=50000, help="window length in bp")
    ap.add_argument("--chroms", default="1-38", help="chromosomes analysed, e.g. 1-38 or 1-5,Z")
    ap.add_argument("--regions", help="comma-separated regions chr:start-end for the gene-level table")
    ap.add_argument("--ann-genes", help="gene table written by annotate (needed with --regions)")
    ap.add_argument(
        "--tier",
        action="append",
        default=[],
        metavar="NAME=GENES_TSV:RESULT_DIR",
        help="gene tables and result directories of build-genes and test-genes (repeatable)",
    )
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.bfile and a.windows and a.snpdir:
        window_min_p(a)
    grm_pcs(a)
    if a.regions and a.ann_genes and a.tier:
        region_genes(a)


if __name__ == "__main__":
    main()
