import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from haplens.cli import main
from haplens.kernel import pval_wchi2


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("haplens")
    main(["simulate", "--out", str(d / "data")])
    return d


def run(work, *args):
    main([str(a) for a in args])


def test_cli_version_and_unknown_command(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip()
    assert main(["no-such-command"]) == 2


def test_saddlepoint_matches_chi_square():
    for q in (3.0, 6.6, 10.8):
        assert pval_wchi2(q, np.array([1.0])) == pytest.approx(stats.chi2.sf(q, 1), rel=0.1)


def test_gene_level_pipeline(work):
    e = work / "data"
    run(
        work,
        "build-genes",
        "--bfile",
        e / "ex",
        "--ann",
        e / "ann_variants.tsv",
        "--genes",
        e / "ann_genes.tsv",
        "--pheno",
        e / "pheno.csv",
        "--tier",
        "A",
        "--out",
        work / "genesA",
    )
    run(
        work,
        "test-genes",
        "--fh",
        work / "genesA_fh.npz",
        "--pheno",
        e / "pheno.csv",
        "--rel",
        e / "grm",
        "--out",
        work / "res_genes",
    )
    truth = json.load(open(e / "truth.json"))
    t1 = pd.read_csv(work / "res_genes" / "T1.csv")
    t0 = pd.read_csv(work / "res_genes" / "T0.csv")
    assert t1[t1.gene_id.isin(truth)].P.min() < 1e-4
    assert (t0.P < 0.05 / len(t0)).sum() == 0
    assert t1[t1.gene_id.isin(truth)].P.median() < t1[~t1.gene_id.isin(truth)].P.median()


def test_window_pipeline(work):
    e = work / "data"
    run(
        work, "make-windows", "--bim", e / "ex.bim", "--size", 50000, "--chroms", 1, "--out", work / "w50.det"
    )
    run(
        work,
        "build-windows",
        "--bfile",
        e / "ex",
        "--blocks",
        e / "windows.det",
        "--pheno",
        e / "pheno.csv",
        "--max-tags",
        4,
        "--min-n",
        30,
        "--out",
        work / "win",
    )
    run(
        work,
        "test-genes",
        "--fh",
        work / "win_fh.npz",
        "--pheno",
        e / "pheno.csv",
        "--rel",
        e / "grm",
        "--out",
        work / "res_win",
    )
    run(
        work,
        "test-windows",
        "--bfile",
        e / "ex",
        "--windows",
        e / "windows.det",
        "--pheno",
        e / "pheno.csv",
        "--rel",
        e / "grm",
        "--chroms",
        1,
        "--perm",
        2,
        "--out",
        work / "kern",
    )
    run(work, "combine-windows", "--kern", work / "kern", "--cls", work / "res_win", "--out", work / "report")
    acat = pd.read_csv(work / "report" / "T1_acat.csv")
    assert {"P_kern", "P_cls", "P_acat", "q_acat"} <= set(acat.columns)
    assert acat.P_acat.between(0, 1).all()
    assert (work / "report" / "window_report.txt").exists()


def test_snp_scan_and_conditional(work):
    e = work / "data"
    run(
        work,
        "snp-scan",
        "--bfile",
        e / "ex",
        "--pheno",
        e / "pheno.csv",
        "--rel",
        e / "grm",
        "--traits",
        "T1",
        "--regions",
        "1:100000-400000",
        "--out",
        work / "snp",
    )
    assert (work / "snp" / "T1_P.npz").exists()
    run(
        work,
        "conditional",
        "--bfile",
        e / "ex",
        "--pheno",
        e / "pheno.csv",
        "--rel",
        e / "grm",
        "--traits",
        "T1",
        "--pairs",
        "1:1399000-1438000=1:1420000",
        "--out",
        work / "cond",
    )
    tab = pd.read_csv(work / "cond" / "conditional.csv")
    assert len(tab) == 1 and 0 < tab.P_conditional.iloc[0] <= 1


def test_check_gene(work):
    e = work / "data"
    run(
        work,
        "check-gene",
        "--fh",
        work / "genesA_fh.npz",
        "--genes-tsv",
        work / "genesA_genes.tsv",
        "--pheno",
        e / "pheno.csv",
        "--rel",
        e / "grm",
        "--bfile",
        e / "ex",
        "--gene-id",
        "G14",
        "--focus",
        "T1",
        "--traits",
        "T1",
        "--perms",
        10,
        "--splits",
        3,
        "--out",
        work / "chk",
    )
    assert os.path.exists(work / "chk" / "summary.txt")
    assert os.path.exists(work / "chk" / "conditional.csv")
