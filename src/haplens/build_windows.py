"""Build haplotype classes of tag SNPs for windows or blocks."""

import argparse, numpy as np, pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens build-windows", description="Build haplotype classes of tag SNPs for windows or blocks."
    )
    ap.add_argument("--bfile", required=True, help="PLINK binary genotype prefix (.bed/.bim/.fam)")
    ap.add_argument("--blocks", required=True, help="block or window table (PLINK .blocks.det layout)")
    ap.add_argument(
        "--pheno",
        required=True,
        help="phenotype table with columns IID, sex, batch, sire, dam and one column per trait",
    )
    ap.add_argument("--maf", type=float, default=0.05, help="minimum minor allele frequency")
    ap.add_argument("--call", type=float, default=0.9, help="minimum call rate")
    ap.add_argument("--min-n", type=int, default=20, help="minimum number of individuals per haplotype class")
    ap.add_argument("--r2", type=float, default=0.8, help="r2 threshold for tag SNP pruning")
    ap.add_argument("--max-tags", type=int, default=6, help="maximum number of tag SNPs per window")
    ap.add_argument("--min-kb", type=float, default=0.0, help="minimum block length in kb")
    ap.add_argument("--out", default="blockfh", help="output directory or file prefix")
    a = ap.parse_args(argv)
    fam = pd.read_csv(a.bfile + ".fam", sep=r"\s+", header=None, dtype=str)
    n_all = len(fam)
    bpl = (n_all + 3) // 4
    bim = pd.read_csv(
        a.bfile + ".bim",
        sep=r"\s+",
        header=None,
        names=["chr", "id", "cm", "pos", "A1", "A2"],
        dtype={"chr": str},
    )
    ph = pd.read_csv(a.pheno, dtype={"IID": str})
    keep = fam[1].isin(ph.IID).values
    iid = fam[1].values[keep]
    n = len(iid)
    bed = np.memmap(a.bfile + ".bed", dtype=np.uint8, mode="r", offset=3, shape=(len(bim), bpl))
    SH = np.array([0, 2, 4, 6], dtype=np.uint8)
    LUT = np.array([2, -1, 1, 0], dtype=np.int8)

    def dosage(rows):
        raw = np.asarray(bed[rows])
        return LUT[(raw[:, :, None] >> SH & 3).reshape(len(rows), -1)[:, :n_all][:, keep]]

    rows = []
    for line in open(a.blocks):
        f = line.split()
        if not f or f[0] == "CHR":
            continue
        rows.append((f[0], int(f[1]), int(f[2]), float(f[3]), int(f[4])))
    B = pd.DataFrame(rows, columns=["chr", "start", "end", "kb", "n_thin"])
    B = B[B.kb >= a.min_kb].reset_index(drop=True)
    print("blocks:", len(B), flush=True)
    chr_idx = {c: np.nonzero(bim.chr.values == c)[0] for c in bim.chr.unique()}
    meta, labs = ([], [])
    for c in sorted(B.chr.unique(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0)):
        if c not in chr_idx:
            continue
        ridx = chr_idx[c]
        ridx = ridx[np.argsort(bim.pos.values[ridx], kind="stable")]
        pos = bim.pos.values[ridx]
        m_c = len(ridx)
        G = np.empty((m_c, n), np.int8)
        call = np.empty(m_c)
        maf = np.empty(m_c)
        for s0 in range(0, m_c, 100000):
            s1 = min(s0 + 100000, m_c)
            g = dosage(ridx[s0:s1])
            G[s0:s1] = g
            ok_ = g >= 0
            call[s0:s1] = ok_.mean(1)
            af = np.maximum(g, 0).sum(1, dtype=np.int64) / np.maximum(2 * ok_.sum(1), 1)
            maf[s0:s1] = np.minimum(af, 1 - af)
        okv = (call >= a.call) & (maf >= a.maf)
        for b in B[B.chr == c].itertuples():
            i0, i1 = (np.searchsorted(pos, b.start), np.searchsorted(pos, b.end, side="right"))
            idx = np.nonzero(okv[i0:i1])[0] + i0
            if len(idx) < 2:
                continue
            X = np.where(G[idx] >= 0, G[idx], np.nan).astype(float)
            mu = np.nanmean(X, 1, keepdims=True)
            X = np.where(np.isnan(X), mu, X)
            X = X - X.mean(1, keepdims=True)
            sd = X.std(1)
            sd[sd == 0] = 1
            X /= sd[:, None]
            C = (X @ X.T / X.shape[1]) ** 2
            kp = []
            for i in np.argsort(-maf[idx]):
                if len(kp) >= a.max_tags:
                    break
                if all((C[i, j] < a.r2 for j in kp)):
                    kp.append(i)
            kp = sorted(kp)
            T = G[idx[kp]]
            ok = (T >= 0).all(0)
            if ok.sum() < 2 * a.min_n:
                continue
            uniq, inv, cnt = np.unique(T[:, ok].T, axis=0, return_inverse=True, return_counts=True)
            inv = inv.ravel()
            kc = np.nonzero(cnt >= a.min_n)[0]
            if len(kc) < 2:
                continue
            order = kc[np.argsort(-cnt[kc])]
            lab = np.full(n, -1, np.int16)
            io = np.nonzero(ok)[0]
            for j, c_ in enumerate(order):
                lab[io[inv == c_]] = j
            meta.append(
                dict(
                    block=f"{c}:{b.start}-{b.end}",
                    chr=c,
                    start=b.start,
                    end=b.end,
                    kb=b.kb,
                    n_snp=len(idx),
                    n_tag=len(kp),
                    n_cls=len(order),
                    n_ind=int((lab >= 0).sum()),
                )
            )
            labs.append(lab)
        print("chr", c, "blocks kept", len(meta), flush=True)
    M = pd.DataFrame(meta)
    M.to_csv(a.out + "_blocks.tsv", sep="\t", index=False)
    np.savez_compressed(
        a.out + "_fh.npz",
        genes=M.block.values,
        fh=np.vstack(labs),
        ctl=np.full((len(labs), n), -1, np.int16),
        iid=iid,
    )
    print(
        "testable blocks:",
        len(M),
        "median kb %.1f" % M.kb.median(),
        "median tags",
        M.n_tag.median(),
        "median classes",
        M.n_cls.median(),
        "bp covered %.1f Mb" % ((M.end - M.start).sum() / 1000000.0),
    )


if __name__ == "__main__":
    main()
