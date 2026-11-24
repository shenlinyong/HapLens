import gzip
import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from haplens.cli import main
from haplens.kernel import acat_combine, pval_wchi2


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("haplens")
    main(["simulate", "--out", str(d / "data")])
    return d


def run(work, *args):
    ret = main([str(a) for a in args])
    assert ret == 0


def test_cli_version_and_unknown_command(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "1.0.0"
    assert main(["no-such-command"]) == 2


def test_saddlepoint_and_acat():
    for q in (3.0, 6.6, 10.8):
        assert pval_wchi2(q, np.array([1.0])) == pytest.approx(stats.chi2.sf(q, 1), rel=0.1)
    assert 0.0 < acat_combine([1e-6, 0.5]) < 5e-6


def test_gene_level_pipeline(work):
    e = work / "data"
    run(
        work, "build-genes", "--bfile", e / "ex", "--ann", e / "ann_variants.tsv",
        "--genes", e / "ann_genes.tsv", "--pheno", e / "pheno.csv", "--tier", "A", "--out", work / "genesA",
    )
    run(
        work, "test-genes", "--fh", work / "genesA_fh.npz", "--pheno", e / "pheno.csv",
        "--rel", e / "grm", "--out", work / "res_genes",
    )
    truth = json.load(open(e / "truth.json"))
    t1 = pd.read_csv(work / "res_genes" / "T1.csv")
    t0 = pd.read_csv(work / "res_genes" / "T0.csv")
    assert t1[t1.gene_id.isin(truth)].P.min() < 1e-4
    assert (t0.P < 0.05 / len(t0)).sum() == 0
    assert t1[t1.gene_id.isin(truth)].P.median() < t1[~t1.gene_id.isin(truth)].P.median()


def test_window_omnibus_and_sliding_pipeline(work):
    e = work / "data"
    run(
        work, "make-windows", "--bim", e / "ex.bim", "--size", 50000, "--step", 25000,
        "--chroms", 1, "--out", work / "w50_step25.det",
    )
    run(
        work, "build-windows", "--bfile", e / "ex", "--blocks", e / "windows.det",
        "--pheno", e / "pheno.csv", "--max-tags", 4, "--min-n", 30, "--out", work / "win",
    )
    run(
        work, "test-genes", "--fh", work / "win_fh.npz", "--pheno", e / "pheno.csv",
        "--rel", e / "grm", "--out", work / "res_win",
    )
    run(
        work, "test-windows", "--bfile", e / "ex", "--windows", e / "windows.det",
        "--pheno", e / "pheno.csv", "--rel", e / "grm", "--chroms", 1,
        "--max-snp", 0, "--whiten", "sym", "--perm", 4, "--out", work / "kern",
    )
    k1 = pd.read_csv(work / "kern" / "T1_1.csv")
    assert {"P", "P_acatv", "P_min_snp", "P_omni"} <= set(k1.columns)
    assert k1.P_omni.between(0, 1).all()
    run(work, "combine-windows", "--kern", work / "kern", "--cls", work / "res_win", "--out", work / "report")
    acat = pd.read_csv(work / "report" / "T1_acat.csv")
    assert {"P_kern", "P_cls", "P_acat", "q_acat", "P_haplens_o", "q_haplens_o"} <= set(acat.columns)
    assert acat.P_haplens_o.between(0, 1).all()
    assert (work / "report" / "window_report.txt").exists()


def test_lbc_and_bench_lbc(work):
    e = work / "data"
    run(
        work, "lbc-test", "--bfile", e / "ex", "--ann", e / "ann_variants.tsv",
        "--genes", e / "ann_genes.tsv", "--pheno", e / "pheno.csv", "--rel", e / "grm",
        "--traits", "T0,T1", "--chroms", 1, "--flank", 40000, "--buffer", 1000,
        "--n-ctrl", 5, "--out", work / "lbc",
    )
    lbc = pd.read_csv(work / "lbc" / "lbc_results.csv")
    assert {"P_code", "P_bg", "P_lbc", "P_rev", "max_r2_code_bg", "bg_r2_explained", "q_code", "q_lbc", "csi", "status"} <= set(lbc.columns)
    truth = json.load(open(e / "truth.json"))
    t1 = lbc[lbc.trait == "T1"]
    assert t1[t1.gene_id.isin(truth)].P_lbc.min() < 1e-3

    run(
        work, "bench-lbc", "--bfile", e / "ex", "--ann", e / "ann_variants.tsv",
        "--genes", e / "ann_genes.tsv", "--pheno", e / "pheno.csv", "--rel", e / "grm",
        "--trait", "T0", "--chroms", 1, "--n-genes", 5, "--pve", "0.04",
        "--flank", 40000, "--buffer", 1000, "--min-hitch-r2", 0.0, "--max-hitch-r2", 1.0,
        "--out", work / "bench_lbc.csv",
    )
    bdf = pd.read_csv(work / "bench_lbc.csv")
    assert len(bdf) > 0 and {"P_fh_class", "P_code_kern", "P_lbc", "P_rev"} <= set(bdf.columns)


def test_phased_genes_python_fallback(work):
    e = work / "data"
    ph = pd.read_csv(e / "pheno.csv")
    ann = pd.read_csv(e / "ann_variants.tsv", sep="\t")
    sub_ann = ann[ann.gene_id.isin(["G0", "G1", "G2"])].sort_values("pos")
    rng = np.random.default_rng(7)
    vcf_path = work / "mini_phased.vcf.gz"
    samples = ph["IID"].astype(str).tolist()
    with gzip.open(vcf_path, "wt") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write("##INFO=<ID=AF,Number=A,Type=Float,Description=\"AF\">\n")
        fh.write("##INFO=<ID=INFO,Number=A,Type=Float,Description=\"INFO\">\n")
        fh.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Phased GT\">\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n")
        for r in sub_ann.itertuples():
            a1 = rng.binomial(1, 0.3, len(samples))
            a2 = rng.binomial(1, 0.3, len(samples))
            af = float((a1.sum() + a2.sum()) / (2 * len(samples)))
            gts = "\t".join(f"{x}|{y}" for x, y in zip(a1, a2))
            fh.write(f"1\t{r.pos}\t1:{r.pos}\tA\tT\t.\tPASS\tAF={af:.4f};INFO=0.96\tGT\t{gts}\n")
    run(
        work, "phased-genes", "--vcf", vcf_path, "--ann", e / "ann_variants.tsv",
        "--pheno", e / "pheno.csv", "--rel", e / "grm", "--traits", "T0",
        "--regions", "1:90000-400000", "--min-n", 15, "--out", work / "phased_out",
    )
    pdf = pd.read_csv(work / "phased_out" / "phased_vs_unphased.csv")
    assert len(pdf) >= 2
    assert {"P_unphased", "P_phased_add", "P_phased_dip", "P_gene_kernel"} <= set(pdf.columns)


def test_power_windows_multiarch(work):
    e = work / "data"
    run(
        work, "power-windows", "--bfile", e / "ex", "--windows", e / "windows.det",
        "--pheno", e / "pheno.csv", "--rel", e / "grm", "--trait", "T0", "--chroms", 1,
        "--n-win", 5, "--arch", "1_causal,3_causal_same,3_causal_opp,2_epistasis", "--pve", "0.04",
        "--out", work / "pow.csv",
    )
    df = pd.read_csv(work / "pow.csv")
    assert set(df.arch.unique()) >= {"null_model", "1_causal", "3_causal_same", "3_causal_opp", "2_epistasis"}
    assert {"P_kern", "P_skat_beta", "P_cls", "P_acatv", "P_haplens_o", "P_min_snp"} <= set(df.columns)


def test_snp_scan_and_conditional(work):
    e = work / "data"
    run(
        work, "snp-scan", "--bfile", e / "ex", "--pheno", e / "pheno.csv",
        "--rel", e / "grm", "--traits", "T1", "--regions", "1:100000-400000", "--out", work / "snp",
    )
    assert (work / "snp" / "T1_P.npz").exists()
    run(
        work, "conditional", "--bfile", e / "ex", "--pheno", e / "pheno.csv",
        "--rel", e / "grm", "--traits", "T1", "--pairs", "1:1399000-1438000=1:1420000", "--out", work / "cond",
    )
    tab = pd.read_csv(work / "cond" / "conditional.csv")
    assert len(tab) == 1 and 0 < tab.P_conditional.iloc[0] <= 1


def test_check_gene_and_run_pipeline(work):
    e = work / "data"
    run(
        work, "check-gene", "--fh", work / "genesA_fh.npz", "--genes-tsv", work / "genesA_genes.tsv",
        "--pheno", e / "pheno.csv", "--rel", e / "grm", "--bfile", e / "ex",
        "--gene-id", "G14", "--focus", "T1", "--traits", "T1", "--perms", 10, "--splits", 3,
        "--out", work / "chk",
    )
    assert os.path.exists(work / "chk" / "summary.txt")
    assert os.path.exists(work / "chk" / "conditional.csv")

    run(
        work, "run", "--bfile", e / "ex", "--pheno", e / "pheno.csv", "--rel", e / "grm",
        "--traits", "T1", "--chroms", 1, "--perm", 2,
        "--ann", e / "ann_variants.tsv", "--genes", e / "ann_genes.tsv", "--out", work / "e2e",
    )
    assert (work / "e2e" / "window_report" / "T1_acat.csv").exists()
    assert (work / "e2e" / "lbc_A" / "lbc_results.csv").exists()


def test_saddlepoint_no_cancellation_near_mean():
    lam = np.concatenate([[200.0, 150.0, 100.0], np.linspace(1.0, 0.01, 800)])
    mu = float(lam.sum())
    for eps in (-1e-6, -1e-8, 0.0, 1e-8, 1e-6):
        p = pval_wchi2(mu * (1.0 + eps), lam)
        assert 0.30 < p < 0.45
