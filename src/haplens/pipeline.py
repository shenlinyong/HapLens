"""End-to-end HapLens workflow: window layer (HapLens-O) and optional gene layer (Bi-LBC)."""

import argparse
import os

from haplens import build_genes, build_windows, combine_windows, lbc_test, make_windows, test_genes, window_kernel


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens run",
        description="Run the complete HapLens multi-scale pipeline (windows + optional gene-level Bi-LBC).",
    )
    ap.add_argument("--bfile", required=True, help="PLINK binary genotype prefix (.bed/.bim/.fam)")
    ap.add_argument("--pheno", required=True, help="phenotype CSV")
    ap.add_argument("--rel", required=True, help="PLINK square GRM prefix (.rel and .rel.id)")
    ap.add_argument("--traits", default=None, help="comma-separated trait names")
    ap.add_argument("--no-sex-traits", default="")
    ap.add_argument("--chroms", default="1-38")
    ap.add_argument("--window-size", type=int, default=50000)
    ap.add_argument("--window-step", type=int, default=0, help="sliding step bp (0 = non-overlapping)")
    ap.add_argument("--perm", type=int, default=10)
    ap.add_argument("--whiten", choices=["cholesky", "sym"], default="cholesky")
    ap.add_argument("--loco-rel", default="")
    ap.add_argument("--ann", default="", help="optional SNP annotation table for gene layer & Bi-LBC")
    ap.add_argument("--genes", default="", help="optional gene table for gene layer & Bi-LBC")
    ap.add_argument("--tier", default="A", choices=["A", "B"])
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    win_det = os.path.join(a.out, "windows.det")
    make_windows.main(
        ["--bim", a.bfile + ".bim", "--size", str(a.window_size), "--step", str(a.window_step),
         "--chroms", a.chroms, "--out", win_det]
    )

    win_fh_prefix = os.path.join(a.out, "win_cls")
    build_windows.main(
        ["--bfile", a.bfile, "--blocks", win_det, "--pheno", a.pheno, "--max-tags", "4", "--min-n", "30",
         "--out", win_fh_prefix]
    )

    cls_dir = os.path.join(a.out, "res_win_cls")
    tg_args = ["--fh", win_fh_prefix + "_fh.npz", "--pheno", a.pheno, "--rel", a.rel, "--out", cls_dir]
    if a.traits:
        tg_args += ["--traits", a.traits]
    if a.no_sex_traits:
        tg_args += ["--no-sex-traits", a.no_sex_traits]
    test_genes.main(tg_args)

    kern_dir = os.path.join(a.out, "res_win_kern")
    tw_args = [
        "--bfile", a.bfile, "--windows", win_det, "--pheno", a.pheno, "--rel", a.rel,
        "--chroms", a.chroms, "--max-snp", "0", "--perm", str(a.perm), "--whiten", a.whiten,
        "--out", kern_dir,
    ]
    if a.traits:
        tw_args += ["--traits", a.traits]
    if a.no_sex_traits:
        tw_args += ["--no-sex-traits", a.no_sex_traits]
    if a.loco_rel:
        tw_args += ["--loco-rel", a.loco_rel]
    window_kernel.main(tw_args)

    rep_dir = os.path.join(a.out, "window_report")
    combine_windows.main(["--kern", kern_dir, "--cls", cls_dir, "--out", rep_dir])

    if a.ann and a.genes:
        gene_prefix = os.path.join(a.out, f"genes_{a.tier}")
        build_genes.main(
            ["--bfile", a.bfile, "--ann", a.ann, "--genes", a.genes, "--pheno", a.pheno,
             "--tier", a.tier, "--out", gene_prefix]
        )
        gene_res = os.path.join(a.out, f"res_genes_{a.tier}")
        g_args = ["--fh", gene_prefix + "_fh.npz", "--pheno", a.pheno, "--rel", a.rel, "--out", gene_res]
        if a.traits:
            g_args += ["--traits", a.traits]
        if a.no_sex_traits:
            g_args += ["--no-sex-traits", a.no_sex_traits]
        test_genes.main(g_args)

        if a.traits:
            lbc_dir = os.path.join(a.out, f"lbc_{a.tier}")
            lbc_args = [
                "--bfile", a.bfile, "--ann", a.ann, "--genes", a.genes, "--pheno", a.pheno,
                "--rel", a.rel, "--traits", a.traits, "--chroms", a.chroms, "--tier", a.tier,
                "--whiten", a.whiten, "--out", lbc_dir,
            ]
            if a.no_sex_traits:
                lbc_args += ["--no-sex-traits", a.no_sex_traits]
            lbc_test.main(lbc_args)


if __name__ == "__main__":
    main()
