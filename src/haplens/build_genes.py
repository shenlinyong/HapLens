"""Build gene-level functional haplotypes and matched neutral control haplotypes."""

import argparse, numpy as np, pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens build-genes",
        description="Build gene-level functional haplotypes and matched neutral control haplotypes.",
    )
    ap.add_argument("--bfile", required=True, help="PLINK binary genotype prefix (.bed/.bim/.fam)")
    ap.add_argument("--ann", required=True, help="SNP annotation table written by annotate")
    ap.add_argument("--genes", required=True, help="gene table written by annotate")
    ap.add_argument(
        "--pheno",
        required=True,
        help="phenotype table with columns IID, sex, batch, sire, dam and one column per trait",
    )
    ap.add_argument(
        "--tier",
        default="A",
        choices=["A", "B"],
        help="variant tier: A (coding and splice-site) or B (A plus regulatory-region variants)",
    )
    ap.add_argument("--maf", type=float, default=0.01, help="minimum minor allele frequency")
    ap.add_argument("--call", type=float, default=0.9, help="minimum call rate")
    ap.add_argument("--min-n", type=int, default=20, help="minimum number of individuals per haplotype class")
    ap.add_argument("--controls", type=int, default=1)
    ap.add_argument(
        "--ctl-windows",
        type=int,
        nargs="+",
        default=[20000, 100000, 500000],
        help="widening search windows (bp) around the gene for neutral control SNPs",
    )
    ap.add_argument("--ctl-tries", type=int, default=15)
    ap.add_argument(
        "--prune-r2",
        type=float,
        default=0.0,
        help="greedy LD pruning of the functional variants of a gene (0 = off)",
    )
    ap.add_argument("--seed", type=int, default=1, help="random seed")
    ap.add_argument("--out", default="fh", help="output directory or file prefix")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    fam = pd.read_csv(a.bfile + ".fam", sep=r"\s+", header=None, dtype=str)
    n_all = len(fam)
    bpl = (n_all + 3) // 4
    bim = pd.read_csv(
        a.bfile + ".bim",
        sep=r"\s+",
        header=None,
        names=["chr", "id", "cm", "pos", "A1", "A2"],
        dtype={"chr": str, "A1": str, "A2": str},
    )
    bim["row"] = np.arange(len(bim))
    ph = pd.read_csv(a.pheno, dtype={"IID": str})
    keep = fam[1].isin(ph.IID).values
    iid = fam[1].values[keep]
    n = len(iid)
    print("individuals with phenotypes:", n, "of", n_all, flush=True)
    bed = np.memmap(a.bfile + ".bed", dtype=np.uint8, mode="r", offset=3, shape=(len(bim), bpl))
    SH = np.array([0, 2, 4, 6], dtype=np.uint8)
    LUT = np.array([2, -1, 1, 0], dtype=np.int8)

    def dosage(rows, alt_is_A1):
        """alt-allele dosage (int8; -1 missing) for the given bim rows, kept individuals only"""
        raw = np.asarray(bed[rows])
        g = (raw[:, :, None] >> SH & 3).reshape(len(rows), -1)[:, :n_all][:, keep]
        d = LUT[g]
        flip = ~alt_is_A1
        if flip.any():
            x = d[flip]
            d[flip] = np.where(x >= 0, 2 - x, -1).astype(np.int8)
        return d

    ann = pd.read_csv(a.ann, sep="\t", dtype={"chr": str})
    tiers = ["A"] if a.tier == "A" else ["A", "B"]
    func = ann[ann.tier.isin(tiers)].copy()
    func_rows_all = set(ann[ann.tier.isin(["A", "B"])].row.values)
    FUNC_ROWS = np.fromiter(func_rows_all, dtype=np.int64)
    genes = pd.read_csv(a.genes, sep="\t", dtype={"chr": str}).set_index("gene_id")

    def classes(G):
        """G: (k, n) int8 dosage; returns per-individual class index (-1 missing/rare) and the class table"""
        ok = (G >= 0).all(0)
        if ok.sum() == 0:
            return (np.full(n, -1, np.int16), [])
        sub = G[:, ok].T
        uniq, inv, cnt = np.unique(sub, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        keepc = np.nonzero(cnt >= a.min_n)[0]
        out = np.full(n, -1, np.int16)
        idx_ok = np.nonzero(ok)[0]
        order = keepc[np.argsort(-cnt[keepc])]
        remap = {c: j for j, c in enumerate(order)}
        for c, j in remap.items():
            out[idx_ok[inv == c]] = j
        return (out, [("".join(map(str, uniq[c])), int(cnt[c])) for c in order])

    gene_out, fh_mat, ctl_mat, cls_rows, meta = ([], [], [], [], [])
    for c in sorted(func.chr.unique(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0)):
        fc = func[func.chr == c]
        rows = np.sort(fc.row.unique())
        ref_alt = fc.drop_duplicates("row").set_index("row")
        G = dosage(rows, ref_alt.loc[rows].alt_is_A1.values.astype(bool))
        gi = {r: i for i, r in enumerate(rows)}
        call = (G >= 0).mean(1)
        af = np.maximum(G, 0).sum(1, dtype=np.int64) / np.maximum(2 * (G >= 0).sum(1), 1)
        maf = np.minimum(af, 1 - af)
        good = {r: call[gi[r]] >= a.call and maf[gi[r]] >= a.maf for r in rows}
        cbc = bim[(bim.chr == c) & (bim.A1.str.len() == 1) & (bim.A2.str.len() == 1)].sort_values("pos")
        cposv = cbc.pos.values
        crow = cbc.row.values
        cfunc = np.isin(crow, FUNC_ROWS)
        for g, sub in fc.groupby("gene_id"):
            r = [x for x in sub.row.unique() if good[x]]
            if not r:
                continue
            Gg = G[[gi[x] for x in r]]
            _, first = np.unique(Gg, axis=0, return_index=True)
            first = np.sort(first)
            r = [r[i] for i in first]
            Gg = Gg[first]
            if a.prune_r2 > 0 and len(r) > 1:
                X = np.where(Gg >= 0, Gg, np.nan).astype(float)
                mu = np.nanmean(X, 1, keepdims=True)
                X = np.where(np.isnan(X), mu, X)
                X = X - X.mean(1, keepdims=True)
                sd = X.std(1)
                sd[sd == 0] = 1
                X = X / sd[:, None]
                C = (X @ X.T / X.shape[1]) ** 2
                kp = []
                for i in np.argsort([-maf[gi[x]] for x in r]):
                    if all((C[i, j] < a.prune_r2 for j in kp)):
                        kp.append(i)
                kp = sorted(kp)
                r = [r[i] for i in kp]
                Gg = Gg[kp]
            fh, tab = classes(Gg)
            if len(tab) < 2:
                continue
            row = dict(
                gene_id=g,
                gene_name=genes.loc[g, "gene_name"],
                chr=c,
                n_var=len(r),
                n_fh=len(tab),
                n_ind=int((fh >= 0).sum()),
                variants=";".join(
                    (f"{ref_alt.loc[x, 'pos']}:{ref_alt.loc[x, 'klass']}:{ref_alt.loc[x, 'aa']}" for x in r)
                ),
            )
            ctl = np.full(n, -1, np.int16)
            ctab = []
            if a.controls:
                best = None
                mv = np.array([maf[gi[x]] for x in r])
                gs, ge = (int(genes.loc[g, "start"]), int(genes.loc[g, "end"]))
                for w in a.ctl_windows:
                    i0, i1 = (np.searchsorted(cposv, gs - w), np.searchsorted(cposv, ge + w, side="right"))
                    sel = np.nonzero(~cfunc[i0:i1])[0] + i0
                    if len(sel) < 2 * len(r):
                        continue
                    Gc = dosage(crow[sel], np.ones(len(sel), bool))
                    callc = (Gc >= 0).mean(1)
                    afc = np.maximum(Gc, 0).sum(1, dtype=np.int64) / np.maximum(2 * (Gc >= 0).sum(1), 1)
                    mafc = np.minimum(afc, 1 - afc)
                    okc = (callc >= a.call) & (mafc >= a.maf)
                    if okc.sum() < len(r):
                        continue
                    Gc = Gc[okc]
                    mafc = mafc[okc]
                    for _try in range(a.ctl_tries):
                        used = np.zeros(len(mafc), bool)
                        pick = []
                        for m0 in mv:
                            cand = np.nonzero(~used & (np.abs(mafc - m0) <= 0.02))[0]
                            if len(cand) == 0:
                                cand = np.nonzero(~used & (np.abs(mafc - m0) <= 0.05))[0]
                            if len(cand) == 0:
                                break
                            j = cand[rng.integers(len(cand))]
                            used[j] = True
                            pick.append(j)
                        if len(pick) != len(r):
                            continue
                        c_, t_ = classes(Gc[pick])
                        if len(t_) < 2:
                            continue
                        score = (
                            abs(len(t_) - len(tab))
                            + 10 * abs(int((c_ >= 0).sum()) - int((fh >= 0).sum())) / n
                        )
                        if best is None or score < best[0]:
                            best = (score, c_, t_)
                    if best is not None and best[0] <= 2.0:
                        break
                if best is not None:
                    ctl, ctab = (best[1], best[2])
            row.update(n_fh_ctl=len(ctab), n_ind_ctl=int((ctl >= 0).sum()))
            gene_out.append(g)
            fh_mat.append(fh)
            ctl_mat.append(ctl)
            meta.append(row)
            for j, (s, cnt) in enumerate(tab):
                cls_rows.append((g, j, s, cnt))
        print("chr", c, "genes so far", len(gene_out), flush=True)
    np.savez_compressed(
        a.out + "_fh.npz", genes=np.array(gene_out), fh=np.vstack(fh_mat), ctl=np.vstack(ctl_mat), iid=iid
    )
    pd.DataFrame(meta).to_csv(a.out + "_genes.tsv", sep="\t", index=False)
    pd.DataFrame(cls_rows, columns=["gene_id", "fh", "genotype", "n"]).to_csv(
        a.out + "_fh_classes.tsv", sep="\t", index=False
    )
    m = pd.DataFrame(meta)
    print(
        "testable genes:",
        len(m),
        " median variants:",
        m.n_var.median(),
        " median FH classes:",
        m.n_fh.median(),
    )


if __name__ == "__main__":
    main()
