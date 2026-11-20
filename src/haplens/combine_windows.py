"""Combine class and kernel tests of windows by ACAT and write the window report."""

import argparse, glob, os, numpy as np, pandas as pd
from scipy import stats


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens combine-windows",
        description="Combine class and kernel tests of windows by ACAT and write the window report.",
    )
    ap.add_argument("--kern", default="kern", help="directory with the kernel results of test-windows")
    ap.add_argument(
        "--cls", default="res_win50t4", help="directory with the class-test results of test-genes"
    )
    ap.add_argument("--out", default="report", help="output directory or file prefix")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    def acat(p1, p2):
        p = np.clip(np.column_stack([p1, p2]), 1e-300, 1 - 1e-16)
        T = np.tan((0.5 - p) * np.pi).mean(1)
        return np.where(
            T > 1000000000000000.0, 1 / (np.pi * np.maximum(T, 1e-300)), 0.5 - np.arctan(T) / np.pi
        )

    def bh(p):
        o = np.argsort(p)
        r = np.empty(len(p))
        r[o] = np.arange(1, len(p) + 1)
        q = p * len(p) / r
        qs = np.minimum.accumulate(q[o][::-1])[::-1]
        out = np.empty(len(p))
        out[o] = np.minimum(qs, 1)
        return out

    rows, allp, L = ([], {}, [])
    tr = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{a.kern}/[A-Z]*_*.csv")})
    for t in tr:
        K = pd.concat([pd.read_csv(f) for f in glob.glob(f"{a.kern}/{t}_*.csv")])
        K["win"] = K.chr.astype(str) + ":" + K.start.astype(str) + "-" + K.end.astype(str)
        fc = f"{a.cls}/{t}.csv"
        if not os.path.exists(fc):
            print("no class result for", t)
            continue
        C = pd.read_csv(fc)[["gene_id", "P", "df", "pve"]].rename(
            columns={"gene_id": "win", "P": "P_cls", "df": "df_cls", "pve": "pve_cls"}
        )
        M = K.rename(columns={"P": "P_kern"}).merge(C, on="win", how="inner")
        M["P_acat"] = acat(M.P_kern.values, M.P_cls.values)
        N = len(M)
        M["q_acat"] = bh(M.P_acat.values)
        M.to_csv(f"{a.out}/{t}_acat.csv", index=False)
        M["trait"] = t
        allp[t] = M.set_index("win").P_acat
        for nm, col in (("kernel", "P_kern"), ("class", "P_cls"), ("ACAT", "P_acat")):
            p = M[col].values
            rows.append(
                dict(
                    trait=t,
                    test=nm,
                    windows=N,
                    bonf=int((p < 0.05 / N).sum()),
                    suggestive=int((p < 1 / N).sum()),
                    fdr10=int((bh(p) < 0.1).sum()),
                    p001=int((p < 0.001).sum()),
                    expected_p001=round(N * 0.001, 1),
                )
            )
        L.append(
            M.sort_values("P_acat").head(5)[
                ["trait", "win", "n_snp", "P_kern", "P_cls", "df_cls", "P_acat", "q_acat"]
            ]
        )
    R = pd.DataFrame(rows)
    P = pd.DataFrame(allp)
    hits = (P < 0.001).sum(axis=1)
    exp = np.mean([(P[c] < 0.001).mean() for c in P.columns]) * len(P.columns)
    with open(f"{a.out}/window_report.txt", "w") as f:
        f.write(
            "== thresholds per trait and test (Bonferroni = 0.05/N, suggestive = 1/N, FDR 10%)\n"
            + R.to_string(index=False)
            + "\n\n"
        )
        f.write("== top 5 windows per trait by ACAT P\n" + pd.concat(L).to_string(index=False) + "\n\n")
        f.write(
            f"== windows with ACAT P<1e-3 in >=3 traits: {(hits >= 3).sum()} (expected by chance if independent: {len(P) * stats.binom.sf(2, len(P.columns), exp / len(P.columns)):.1f}); in >=4 traits: {(hits >= 4).sum()}\n"
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
