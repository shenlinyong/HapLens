"""Combine window tests (kernel, ACAT-V and haplotype-class) by ACAT and write the window report."""

import argparse
import glob
import os
import numpy as np
import pandas as pd
from scipy import stats

from haplens.kernel import acat_matrix


def _bh(p):
    p = np.asarray(p, dtype=np.float64)
    o = np.argsort(p)
    r = np.empty(len(p))
    r[o] = np.arange(1, len(p) + 1)
    q = p * len(p) / r
    qs = np.minimum.accumulate(q[o][::-1])[::-1]
    out = np.empty(len(p))
    out[o] = np.minimum(qs, 1.0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens combine-windows",
        description="Combine kernel, ACAT-V and haplotype-class tests of windows by ACAT.",
    )
    ap.add_argument("--kern", default="kern", help="directory with results of test-windows")
    ap.add_argument("--cls", default="res_win50t4", help="directory with class-test results of test-genes")
    ap.add_argument("--out", default="report", help="output directory")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    rows, allp, allp_o, L = ([], {}, {}, [])
    files = sorted(glob.glob(f"{a.kern}/[A-Z]*_*.csv"))
    tr = sorted({os.path.basename(f).split("_")[0] for f in files if not os.path.basename(f).startswith("summary_")})
    for t in tr:
        kfiles = [f for f in glob.glob(f"{a.kern}/{t}_*.csv") if not os.path.basename(f).startswith("summary_")]
        K = pd.concat([pd.read_csv(f) for f in kfiles], ignore_index=True)
        K["win"] = K.chr.astype(str) + ":" + K.start.astype(str) + "-" + K.end.astype(str)
        fc = f"{a.cls}/{t}.csv"
        if not os.path.exists(fc):
            print("no class result for", t)
            continue
        C = pd.read_csv(fc)[["gene_id", "P", "df", "pve"]].rename(
            columns={"gene_id": "win", "P": "P_cls", "df": "df_cls", "pve": "pve_cls"}
        )
        M = K.rename(columns={"P": "P_kern"}).merge(C, on="win", how="inner")
        M["P_acat"] = acat_matrix(np.column_stack([M.P_kern.values, M.P_cls.values]))
        N = len(M)
        M["q_acat"] = _bh(M.P_acat.values)

        has_acatv = "P_acatv" in M.columns
        if has_acatv:
            M["P_haplens_o"] = acat_matrix(
                np.column_stack([M.P_kern.values, M.P_acatv.values, M.P_cls.values])
            )
            M["q_haplens_o"] = _bh(M.P_haplens_o.values)

        M.to_csv(f"{a.out}/{t}_acat.csv", index=False)
        M["trait"] = t
        allp[t] = M.set_index("win").P_acat
        if has_acatv:
            allp_o[t] = M.set_index("win").P_haplens_o

        test_cols = [("kernel", "P_kern"), ("class", "P_cls"), ("ACAT", "P_acat")]
        if has_acatv:
            test_cols += [("ACAT-V", "P_acatv"), ("HapLens-O", "P_haplens_o")]
        for nm, col in test_cols:
            p = M[col].values
            rows.append(
                dict(
                    trait=t,
                    test=nm,
                    windows=N,
                    bonf=int((p < 0.05 / N).sum()),
                    suggestive=int((p < 1.0 / N).sum()),
                    fdr10=int((_bh(p) < 0.1).sum()),
                    p001=int((p < 0.001).sum()),
                    expected_p001=round(N * 0.001, 1),
                )
            )
        show_cols = ["trait", "win", "n_snp", "P_kern", "P_cls", "df_cls", "P_acat", "q_acat"]
        if has_acatv:
            show_cols += ["P_acatv", "P_haplens_o", "q_haplens_o"]
        sort_col = "P_haplens_o" if has_acatv else "P_acat"
        L.append(M.sort_values(sort_col).head(5)[show_cols])

    R = pd.DataFrame(rows)
    P = pd.DataFrame(allp)
    hits = (P < 0.001).sum(axis=1)
    exp = np.mean([(P[c] < 0.001).mean() for c in P.columns]) * len(P.columns) if len(P.columns) else 0.0
    if len(P.columns) > 1:
        multi_acat = pd.DataFrame({"win": P.index, "P_multi_acat": acat_matrix(P.values)})
        if allp_o:
            Po = pd.DataFrame(allp_o)
            multi_acat["P_multi_haplens_o"] = acat_matrix(Po.values)
        multi_acat.to_csv(f"{a.out}/multi_trait_acat.csv", index=False)

    with open(f"{a.out}/window_report.txt", "w") as f:
        f.write(
            "== thresholds per trait and test (Bonferroni = 0.05/N, suggestive = 1/N, FDR 10%)\n"
            + R.to_string(index=False)
            + "\n\n"
        )
        f.write("== top 5 windows per trait\n" + pd.concat(L).to_string(index=False) + "\n\n")
        f.write(
            f"== windows with ACAT P<1e-3 in >=3 traits: {(hits >= 3).sum()} "
            f"(expected by chance if independent: {len(P) * stats.binom.sf(2, len(P.columns), exp / max(len(P.columns), 1)):.1f}); "
            f"in >=4 traits: {(hits >= 4).sum()}\n"
        )
        f.write(
            pd.DataFrame(
                {
                    "traits_hit": hits[hits >= 3],
                    "traits": [",".join(P.columns[(P.loc[w] < 0.001).values]) for w in hits[hits >= 3].index],
                }
            )
            .sort_values("traits_hit", ascending=False)
            .head(20)
            .to_string()
            + "\n"
        )
    print(open(f"{a.out}/window_report.txt").read())


if __name__ == "__main__":
    main()
