"""Annotate SNPs against a gene model and classify them by predicted consequence."""

import gzip, argparse, re, collections
import numpy as np, pandas as pd
from haplens.utils import chrom_list


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens annotate",
        description="Annotate SNPs against a gene model and classify them by predicted consequence.",
    )
    ap.add_argument("--bim", required=True)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", default="ann", help="output directory or file prefix")
    ap.add_argument("--chroms", default="1-39", help="chromosomes analysed, e.g. 1-38 or 1-5,Z")
    ap.add_argument("--upstream", type=int, default=2000)
    a = ap.parse_args(argv)

    CHROMS = chrom_list(a.chroms)
    BASES = "TCAG"
    AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
    CODON = {
        b1 + b2 + b3: AAS[i]
        for i, (b1, b2, b3) in enumerate(((x, y, z) for x in BASES for y in BASES for z in BASES))
    }
    COMP = str.maketrans("ACGT", "TGCA")

    def rc(s):
        return s.translate(COMP)[::-1]

    op = gzip.open if a.gtf.endswith(".gz") else open
    tx = {}
    attr_re = re.compile('(\\w+) "([^"]*)"')
    with op(a.gtf, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if f[0] not in CHROMS or f[2] not in ("transcript", "exon", "CDS", "stop_codon"):
                continue
            at = dict(attr_re.findall(f[8]))
            if at.get("gene_biotype") != "protein_coding" or at.get("transcript_biotype") != "protein_coding":
                continue
            tid = at["transcript_id"]
            t = tx.setdefault(
                tid,
                dict(
                    chr=f[0],
                    strand=f[6],
                    gene=at["gene_id"],
                    name=at.get("gene_name", at["gene_id"]),
                    exon=[],
                    cds=[],
                    stop=[],
                    canon=False,
                    phase0=None,
                ),
            )
            s, e = (int(f[3]), int(f[4]))
            if f[2] == "transcript":
                t["canon"] = 'tag "Ensembl_canonical"' in f[8]
                t["start"], t["end"] = (s, e)
            elif f[2] == "exon":
                t["exon"].append((s, e))
            elif f[2] == "CDS":
                t["cds"].append((s, e, f[7]))
            elif f[2] == "stop_codon":
                t["stop"].append((s, e))
    tx = {k: v for k, v in tx.items() if v["cds"]}
    print("protein-coding transcripts", len(tx), flush=True)
    genes = collections.defaultdict(list)
    for k, v in tx.items():
        genes[v["gene"]].append(k)
    gene_rows = []
    for g, ts in genes.items():
        v = tx[ts[0]]
        gene_rows.append(
            (
                g,
                v["name"],
                v["chr"],
                min((tx[t]["start"] for t in ts)),
                max((tx[t]["end"] for t in ts)),
                v["strand"],
                len(ts),
            )
        )
    pd.DataFrame(gene_rows, columns=["gene_id", "gene_name", "chr", "start", "end", "strand", "n_tx"]).to_csv(
        a.out + "_genes.tsv", sep="\t", index=False
    )
    print("protein-coding genes", len(genes), flush=True)
    bim = pd.read_csv(
        a.bim,
        sep=r"\s+",
        header=None,
        names=["chr", "id", "cm", "pos", "A1", "A2"],
        dtype={"chr": str, "id": str, "A1": str, "A2": str},
    )
    bim["row"] = np.arange(len(bim))
    bim = bim[
        bim.chr.isin(CHROMS)
        & (bim.A1.str.len() == 1)
        & (bim.A2.str.len() == 1)
        & bim.A1.isin(list("ACGT"))
        & bim.A2.isin(list("ACGT"))
    ]
    print("SNPs on selected chromosomes", len(bim), flush=True)
    PRI = {
        "stop_gained": 9,
        "start_lost": 9,
        "splice_site": 9,
        "stop_lost": 8,
        "missense": 7,
        "splice_region": 4,
        "utr5": 3,
        "utr3": 3,
        "upstream": 2,
        "synonymous": 1,
        "stop_retained": 1,
    }
    TIER = {
        "stop_gained": "A",
        "start_lost": "A",
        "splice_site": "A",
        "stop_lost": "A",
        "missense": "A",
        "splice_region": "B",
        "utr5": "B",
        "utr3": "B",
        "upstream": "B",
        "synonymous": "N",
        "stop_retained": "N",
    }

    def annotate_chrom(c, seq, snps):
        """snps: DataFrame of this chromosome sorted by pos; seq: reference sequence (str, 1-based via seq[pos-1])."""
        P = snps.pos.values
        A1 = snps.A1.values
        A2 = snps.A2.values
        ref = np.array([seq[p - 1] if 0 < p <= len(seq) else "N" for p in P])
        ok = (ref == A1) | (ref == A2)
        alt = np.where(ref == A1, A2, A1)
        best = {}

        def put(i, gene, klass, aa, canon):
            k = (i, gene)
            pr = PRI[klass]
            cur = best.get(k)
            if cur is None or pr > cur[0] or (pr == cur[0] and canon and (not cur[3])):
                best[k] = (pr, klass, aa, canon)

        for tid, t in tx.items():
            if t["chr"] != c:
                continue
            strand = t["strand"]
            g = t["gene"]
            lo = max(t["start"] - a.upstream, 0) if strand == "+" else t["start"]
            hi = t["end"] if strand == "+" else t["end"] + a.upstream
            i0, i1 = (np.searchsorted(P, lo), np.searchsorted(P, hi, side="right"))
            if i0 >= i1:
                continue
            idx = np.arange(i0, i1)
            idx = idx[ok[idx]]
            if len(idx) == 0:
                continue
            S = P[idx]
            cds = sorted(t["cds"], key=lambda x: x[0], reverse=strand == "-")
            stop = sorted(t["stop"], key=lambda x: x[0], reverse=strand == "-")
            coding = [(s, e) for s, e, _ in cds] + stop
            phase0 = int(cds[0][2]) if cds[0][2].isdigit() else 0
            cstart = min((s for s, e in coding))
            cend = max((e for s, e in coding))
            parts = [seq[s - 1 : e] for s, e in sorted(coding, key=lambda x: x[0], reverse=strand == "-")]
            cseq = "".join(parts) if strand == "+" else "".join((rc(p) for p in parts))
            order = sorted(coding, key=lambda x: x[0], reverse=strand == "-")
            cum = np.cumsum([0] + [e - s + 1 for s, e in order])
            ncds = int(sum((e - s + 1 for s, e, _ in cds)))
            inc = np.zeros(len(S), bool)
            for j, (s, e) in enumerate(order):
                inc |= (S >= s) & (S <= e)
            for q in np.nonzero(inc)[0]:
                pos = int(S[q])
                i = idx[q]
                for j, (s, e) in enumerate(order):
                    if s <= pos <= e:
                        off = int(cum[j] + (pos - s if strand == "+" else e - pos))
                        break
                cpos = off - phase0
                if cpos < 0:
                    continue
                ci = cpos // 3
                cp = cpos % 3
                codon = cseq[phase0 + 3 * ci : phase0 + 3 * ci + 3]
                if len(codon) < 3 or "N" in codon:
                    continue
                b_alt = alt[i] if strand == "+" else alt[i].translate(COMP)
                if codon[cp] != (ref[i] if strand == "+" else ref[i].translate(COMP)):
                    continue
                ncodon = codon[:cp] + b_alt + codon[cp + 1 :]
                ra, aa_ = (CODON[codon], CODON[ncodon])
                if off >= ncds:
                    klass = "stop_retained" if aa_ == "*" else "stop_lost"
                elif ci == 0 and phase0 == 0 and (ra == "M") and (aa_ != "M"):
                    klass = "start_lost"
                elif aa_ == ra:
                    klass = "synonymous"
                elif aa_ == "*":
                    klass = "stop_gained"
                else:
                    klass = "missense"
                put(i, g, klass, f"{ra}{ci + 1}{aa_}", t["canon"])
            for s, e in t["exon"]:
                m = (S >= s) & (S <= e) & ~inc & ((S < cstart) | (S > cend))
                for q in np.nonzero(m)[0]:
                    left = S[q] < cstart
                    put(idx[q], g, "utr5" if left == (strand == "+") else "utr3", "", t["canon"])
            ex = sorted(t["exon"])
            for (s1, e1), (s2, e2) in zip(ex[:-1], ex[1:]):
                for x0, x1, klass in (
                    (e1 + 1, e1 + 2, "splice_site"),
                    (s2 - 2, s2 - 1, "splice_site"),
                    (e1 - 2, e1, "splice_region"),
                    (s2, s2 + 2, "splice_region"),
                    (e1 + 3, e1 + 8, "splice_region"),
                    (s2 - 8, s2 - 3, "splice_region"),
                ):
                    m = (S >= x0) & (S <= x1)
                    for q in np.nonzero(m)[0]:
                        put(idx[q], g, klass, "", t["canon"])
            pl, ph = (
                (t["start"] - a.upstream, t["start"] - 1)
                if strand == "+"
                else (t["end"] + 1, t["end"] + a.upstream)
            )
            m = (S >= pl) & (S <= ph)
            for q in np.nonzero(m)[0]:
                put(idx[q], g, "upstream", "", t["canon"])
        rows = []
        for (i, g), (pr, klass, aa, canon) in best.items():
            rows.append(
                (
                    c,
                    int(P[i]),
                    snps.id.values[i],
                    int(snps.row.values[i]),
                    ref[i],
                    alt[i],
                    A1[i],
                    g,
                    gene_name[g],
                    klass,
                    TIER[klass],
                    aa,
                )
            )
        return rows

    gene_name = {g: tx[ts[0]]["name"] for g, ts in genes.items()}
    rows = []
    opf = gzip.open if a.fasta.endswith(".gz") else open
    cur, buf = (None, [])

    def flush(cur, buf):
        if cur in CHROMS:
            sub = bim[bim.chr == cur].sort_values("pos")
            if len(sub):
                r = annotate_chrom(cur, "".join(buf).upper(), sub)
                rows.extend(r)
                print("chr", cur, "SNPs", len(sub), "annotated", len(r), flush=True)

    with opf(a.fasta, "rt") as fh:
        for line in fh:
            if line[0] == ">":
                flush(cur, buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    flush(cur, buf)
    out = pd.DataFrame(
        rows,
        columns=[
            "chr",
            "pos",
            "id",
            "row",
            "ref",
            "alt",
            "A1",
            "gene_id",
            "gene_name",
            "klass",
            "tier",
            "aa",
        ],
    )
    out["alt_is_A1"] = out.alt == out.A1
    out.drop(columns="A1").to_csv(a.out + "_variants.tsv", sep="\t", index=False)
    print(out.groupby(["tier", "klass"]).size())


if __name__ == "__main__":
    main()
