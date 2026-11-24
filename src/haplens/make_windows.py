"""Write fixed-size (non-overlapping or sliding) windows that cover each chromosome up to its last SNP."""

import argparse
import pandas as pd
from haplens.utils import chrom_list


def make_windows(bim_path, size, chroms, step=0):
    step = int(step) if step and step > 0 else int(size)
    bim = pd.read_csv(
        bim_path, sep=r"\s+", header=None, names=["chr", "id", "cm", "pos", "A1", "A2"], dtype={"chr": str}
    )
    last = bim.groupby("chr").pos.max()
    rows = []
    for c in chrom_list(chroms):
        if c not in last.index:
            continue
        rows += [(c, s, s + size - 1, size // 1000, 0, "W") for s in range(1, int(last[c]) + 1, step)]
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="haplens make-windows", description=__doc__)
    ap.add_argument("--bim", required=True, help="PLINK .bim file")
    ap.add_argument("--size", type=int, default=50000, help="window length in bp")
    ap.add_argument("--step", type=int, default=0, help="sliding step in bp (0 = non-overlapping)")
    ap.add_argument("--chroms", default="1-38", help="chromosomes analysed, e.g. 1-38 or 1-5,Z")
    ap.add_argument("--out", required=True, help="output window table")
    a = ap.parse_args(argv)
    rows = make_windows(a.bim, a.size, a.chroms, step=a.step)
    with open(a.out, "w") as fh:
        fh.write(" CHR BP1 BP2 KB NSNPS SNPS\n")
        fh.write("".join(f"{c} {s} {e} {kb} {n} {tag}\n" for c, s, e, kb, n, tag in rows))
    step_lbl = a.step if a.step > 0 else a.size
    print(f"wrote {len(rows)} windows of {a.size} bp (step {step_lbl} bp) to {a.out}")


if __name__ == "__main__":
    main()
