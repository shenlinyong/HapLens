"""Phased gene-level haplotype association and comparison with unphased genotype combinations.

Reads phased VCF/BCF files (phased '|' GT calls from GLIMPSE2, Beagle or SHAPEIT; uses bcftools
for fast indexed region queries when available, with a pure-Python gzipped VCF fallback).
Evaluates each gene under the same whitened mixed model using four representations:
  1) Unphased 0/1/2 multilocus genotype combination (diplotype) F-test
  2) Phased additive haplotype dosage F-test (two haplotypes per individual; reduces df)
  3) Phased diplotype F-test (unordered haplotype pairs; distinguishes cis vs trans phase)
  4) 1-df gene coding relatedness-kernel score test
Optionally reports imputation INFO quality stratified by MAF bin.
"""

import argparse
import gzip
import io
import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from scipy import stats

from haplens.kernel import dual_eigvalsh, pval_wchi2
from haplens.model import build_design_matrix, build_random_matrices, load_grm, reml_fit, scan, whiten


def _parse_gt_pair(gt_str):
    sep = "|" if "|" in gt_str else ("/" if "/" in gt_str else None)
    if sep is None:
        return -1, -1
    a, b = gt_str.split(sep, 1)
    a1 = int(a) if a in ("0", "1") else -1
    a2 = int(b) if b in ("0", "1") else -1
    return a1, a2


def _read_vcf_samples(vcf_path):
    if shutil.which("bcftools"):
        return subprocess.check_output(["bcftools", "query", "-l", vcf_path], text=True).strip().splitlines()
    op = gzip.open if vcf_path.endswith(".gz") else open
    with op(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                return line.rstrip("\n").split("\t")[9:]
    raise RuntimeError(f"No #CHROM header found in {vcf_path}")


def _fetch_phased_region(vcf_path, chrom, start, end, target_positions=None, min_maf=0.01):
    """Return (pos, af, info, H1, H2) for biallelic sites in chrom:start-end."""
    tset = set(target_positions) if target_positions is not None else None
    pos_l, af_l, inf_l, h1_l, h2_l = [], [], [], [], []
    region_str = f"{chrom}:{start}-{end}"

    if shutil.which("bcftools") and (os.path.exists(vcf_path + ".csi") or os.path.exists(vcf_path + ".tbi")):
        cmd = ["bcftools", "query", "-r", region_str, "-f", "%CHROM\t%POS\t%INFO/AF\t%INFO/INFO[\t%GT]\n", vcf_path]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        for line in proc.stdout:
            parts = line.rstrip("\n").split("\t")
            pos = int(parts[1])
            if tset is not None and pos not in tset:
                continue
            gts = parts[4:]
            a1 = np.empty(len(gts), dtype=np.int8)
            a2 = np.empty(len(gts), dtype=np.int8)
            for idx, g in enumerate(gts):
                a1[idx], a2[idx] = _parse_gt_pair(g)
            ok = (a1 >= 0) & (a2 >= 0)
            af = float((a1[ok].sum() + a2[ok].sum()) / max(2 * ok.sum(), 1))
            if parts[2] not in (".", ""):
                try:
                    af = float(parts[2].split(",")[0])
                except ValueError:
                    pass
            maf = min(af, 1.0 - af)
            if maf < min_maf:
                continue
            info_v = np.nan
            if parts[3] not in (".", ""):
                try:
                    info_v = float(parts[3].split(",")[0])
                except ValueError:
                    pass
            pos_l.append(pos)
            af_l.append(af)
            inf_l.append(info_v)
            h1_l.append(a1)
            h2_l.append(a2)
        proc.wait()
    else:
        op = gzip.open if vcf_path.endswith(".gz") else open
        with op(vcf_path, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if parts[0] != str(chrom):
                    continue
                pos = int(parts[1])
                if pos < start or pos > end:
                    continue
                if tset is not None and pos not in tset:
                    continue
                info_dict = {}
                for kv in parts[7].split(";"):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        info_dict[k] = v
                gts = [x.split(":", 1)[0] for x in parts[9:]]
                a1 = np.empty(len(gts), dtype=np.int8)
                a2 = np.empty(len(gts), dtype=np.int8)
                for idx, g in enumerate(gts):
                    a1[idx], a2[idx] = _parse_gt_pair(g)
                ok = (a1 >= 0) & (a2 >= 0)
                af = float((a1[ok].sum() + a2[ok].sum()) / max(2 * ok.sum(), 1))
                maf = min(af, 1.0 - af)
                if maf < min_maf:
                    continue
                info_v = float(info_dict["INFO"].split(",")[0]) if "INFO" in info_dict else np.nan
                pos_l.append(pos)
                af_l.append(af)
                inf_l.append(info_v)
                h1_l.append(a1)
                h2_l.append(a2)

    if not pos_l:
        return np.array([], int), np.array([]), np.array([]), np.empty((0, 0), np.int8), np.empty((0, 0), np.int8)
    return np.array(pos_l, int), np.array(af_l), np.array(inf_l), np.vstack(h1_l), np.vstack(h2_l)


def build_phased_additive_matrix(h1, h2, min_chrom=20):
    """Build (n, q) additive phased haplotype dosage matrix (0/1/2 copies of each retained haplotype)."""
    n = h1.shape[1]
    both = np.vstack([h1.T, h2.T])  # (2n, k_var)
    ok = (both >= 0).all(1)
    if ok.sum() < 2 * min_chrom:
        return np.zeros((n, 0)), 0
    _, inv, cnt = np.unique(both[ok], axis=0, return_inverse=True, return_counts=True)
    keep_h = np.nonzero(cnt >= min_chrom)[0]
    if len(keep_h) < 2:
        return np.zeros((n, 0)), int(len(keep_h))
    keep_h = keep_h[np.argsort(-cnt[keep_h])]
    others = keep_h[1:]
    hap_id = np.full(2 * n, -1, dtype=np.int32)
    hap_id[ok] = inv
    h1_id, h2_id = hap_id[:n], hap_id[n:]
    cols = [((h1_id == c).astype(float) + (h2_id == c).astype(float)) for c in others]
    is_rare = (~np.isin(h1_id, keep_h)).astype(float) + (~np.isin(h2_id, keep_h)).astype(float)
    if is_rare.sum() > 0:
        cols.append(is_rare)
    return np.column_stack(cols), int(len(keep_h))


def build_unphased_diplotype_labels(dos, min_n=20):
    n = dos.shape[1]
    ok = (dos >= 0).all(0)
    lab = np.full(n, -1, dtype=np.int16)
    if ok.sum() < 2 * min_n:
        return lab, 0
    _, inv, cnt = np.unique(dos[:, ok].T, axis=0, return_inverse=True, return_counts=True)
    kc = np.nonzero(cnt >= min_n)[0]
    if len(kc) < 2:
        return lab, int(len(kc))
    io = np.nonzero(ok)[0]
    for j, c_ in enumerate(kc[np.argsort(-cnt[kc])]):
        lab[io[inv == c_]] = j
    return lab, int(len(kc))


def build_phased_diplotype_labels(h1, h2, min_n=20):
    n = h1.shape[1]
    ok = ((h1 >= 0).all(0)) & ((h2 >= 0).all(0))
    lab = np.full(n, -1, dtype=np.int16)
    if ok.sum() < 2 * min_n:
        return lab, 0
    idx_ok = np.nonzero(ok)[0]
    pairs = []
    for i in idx_ok:
        t1 = tuple(h1[:, i].tolist())
        t2 = tuple(h2[:, i].tolist())
        pairs.append((t1, t2) if t1 <= t2 else (t2, t1))
    vc = pd.Series(pairs).value_counts()
    keep_p = vc[vc >= min_n].index.tolist()
    if len(keep_p) < 2:
        return lab, int(len(keep_p))
    pmap = {p: j for j, p in enumerate(keep_p)}
    for i, p in zip(idx_ok, pairs):
        if p in pmap:
            lab[i] = pmap[p]
    return lab, int(len(keep_p))


def _ftest_design(D, Z_mat, Z_cov=None):
    if Z_mat.shape[1] == 0:
        return np.nan, 0, np.nan
    A0 = D["Xt"] if Z_cov is None else np.hstack([D["Xt"], D["W"] @ Z_cov])
    A1 = np.hstack([A0, D["W"] @ Z_mat])
    c0, _, rk0, _ = np.linalg.lstsq(A0, D["yt"], rcond=None)
    rss0 = float(((D["yt"] - A0 @ c0) ** 2).sum())
    c1, _, rk1, _ = np.linalg.lstsq(A1, D["yt"], rcond=None)
    rss1 = float(((D["yt"] - A1 @ c1) ** 2).sum())
    q = int(rk1 - rk0)
    dfd = int(D["n"] - rk1)
    if q <= 0 or dfd <= 0 or rss1 <= 0:
        return np.nan, 0, np.nan
    F = ((rss0 - rss1) / q) / (rss1 / dfd)
    return float(stats.f.sf(F, q, dfd)), q, float(1.0 - rss1 / rss0)


def _gene_kernel_p(D, dos):
    X = np.where(dos >= 0, dos, np.nan).astype(float)
    mu = np.nanmean(X, 1, keepdims=True)
    X = np.where(np.isnan(X), mu, X) - mu
    sd = X.std(1)
    keep = sd > 1e-8
    if not keep.any():
        return 1.0
    Z = (X[keep] / sd[keep, None]).T
    Zt = D["W"] @ Z
    B = Zt - D["Xt"] @ (D["Xpinv"] @ Zt)
    lam = dual_eigvalsh(B)
    q = float(((B.T @ D["r0"]) ** 2).sum()) / D["s2"]
    return pval_wchi2(q, lam)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens phased-genes",
        description="Compare phased additive haplotypes, phased diplotypes, unphased combinations and coding kernel.",
    )
    ap.add_argument("--vcf", required=True, help="phased VCF/BCF file (.vcf.gz or .bcf)")
    ap.add_argument("--ann", required=True, help="SNP annotation table (ann_variants.tsv)")
    ap.add_argument("--pheno", required=True, help="phenotype CSV")
    ap.add_argument("--rel", required=True, help="PLINK square GRM prefix")
    ap.add_argument("--traits", required=True, help="comma-separated trait names")
    ap.add_argument("--regions", required=True, help="comma-separated regions chr:start-end")
    ap.add_argument("--no-sex-traits", default="")
    ap.add_argument("--tier", default="A", choices=["A", "B"])
    ap.add_argument("--maf", type=float, default=0.045)
    ap.add_argument("--min-n", type=int, default=20, help="min birds for diplotype / min chromosomes for phased hap")
    ap.add_argument("--audit-info", action="store_true", help="export MAF-stratified VCF INFO quality audit")
    ap.add_argument("--out", default="phased_out")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    no_sex = set(filter(None, a.no_sex_traits.split(",")))
    vcf_samples = _read_vcf_samples(a.vcf)
    vpos = {}
    for idx, s in enumerate(vcf_samples):
        vpos[s] = idx
        if "_" in s:
            vpos[s.split("_")[-1]] = idx
            vpos[s.split("_")[0]] = idx

    ph = pd.read_csv(a.pheno, dtype={"IID": str}).set_index("IID")
    R, rpos, _ = load_grm(a.rel)
    traits = a.traits.split(",")
    nulls = {}
    for t in traits:
        mask = ph[t].notna() & ph.index.isin(rpos.keys()) & ph.index.isin(vpos.keys())
        if "sex" in ph.columns:
            mask &= ph["sex"].notna()
        d = ph[mask]
        ids = d.index.values
        m_ = len(ids)
        y = d[t].values.astype(float)
        y = (y - y.mean()) / max(y.std(), 1e-12)
        X = build_design_matrix(d, t, no_sex_traits=no_sex)
        mats, _ = build_random_matrices(d, R, rpos, random_spec="grm,dam")
        _, V = reml_fit(y, X, mats)
        W = whiten(V)
        yt = W @ y
        Xt = W @ X
        Xpinv = np.linalg.pinv(Xt)
        r0 = yt - Xt @ (Xpinv @ yt)
        s2 = float(r0 @ r0) / max(m_ - Xt.shape[1], 1)
        nulls[t] = dict(
            ids=ids, vidx=np.array([vpos[i] for i in ids]),
            W=W, yt=yt, Xt=Xt, Xpinv=Xpinv, r0=r0, s2=s2, n=m_,
        )

    ann = pd.read_csv(a.ann, sep="	", dtype={"chr": str})
    tiers = ["A"] if a.tier == "A" else ["A", "B"]
    ann_t = ann[ann.tier.isin(tiers)].copy()

    regions = []
    for reg in [x.strip() for x in a.regions.split(",") if x.strip()]:
        c, se = reg.split(":")
        s, e = map(int, se.split("-"))
        regions.append((c, s, e))

    rows = []
    info_dfs = []
    for c, s, e in regions:
        sub_ann = ann_t[(ann_t.chr == c) & (ann_t.pos >= s) & (ann_t.pos <= e)]
        tpos = sorted(sub_ann.pos.unique().tolist())
        p_arr, af_arr, inf_arr, H1, H2 = _fetch_phased_region(a.vcf, c, s, e, target_positions=tpos, min_maf=a.maf)
        pmap = {int(p): i for i, p in enumerate(p_arr)}
        for (gid, gname), grp in sub_ann.groupby(["gene_id", "gene_name"]):
            p_idxs = [pmap[int(p)] for p in sorted(grp.pos.unique()) if int(p) in pmap]
            if not p_idxs:
                continue
            mean_info = float(np.nanmean(inf_arr[p_idxs])) if np.isfinite(inf_arr[p_idxs]).any() else np.nan
            min_info = float(np.nanmin(inf_arr[p_idxs])) if np.isfinite(inf_arr[p_idxs]).any() else np.nan
            for t, D in nulls.items():
                h1 = H1[np.ix_(p_idxs, D["vidx"])]
                h2 = H2[np.ix_(p_idxs, D["vidx"])]
                dos = h1 + h2

                lab_u, ncls_u = build_unphased_diplotype_labels(dos, min_n=a.min_n)
                res_u = scan(D["yt"], D["Xt"], D["W"], lab_u[None, :], want_effects=False)[0] if ncls_u >= 2 else None

                Z_pa, nhap_pa = build_phased_additive_matrix(h1, h2, min_chrom=a.min_n)
                p_pa, df_pa, pve_pa = _ftest_design(D, Z_pa)

                lab_pd, ncls_pd = build_phased_diplotype_labels(h1, h2, min_n=a.min_n)
                res_pd = scan(D["yt"], D["Xt"], D["W"], lab_pd[None, :], want_effects=False)[0] if ncls_pd >= 2 else None

                p_gk = _gene_kernel_p(D, dos)

                rows.append(
                    dict(
                        chr=c, gene_id=gid, gene=gname, trait=t, n_var=len(p_idxs),
                        mean_INFO=mean_info, min_INFO=min_info,
                        unphased_cls=ncls_u, unphased_df=res_u["df"] if res_u else np.nan,
                        P_unphased=res_u["P"] if res_u else np.nan,
                        phased_haps=nhap_pa, phased_add_df=df_pa, P_phased_add=p_pa, PVE_phased_add=pve_pa,
                        phased_dip_cls=ncls_pd, phased_dip_df=res_pd["df"] if res_pd else np.nan,
                        P_phased_dip=res_pd["P"] if res_pd else np.nan,
                        P_gene_kernel=p_gk,
                    )
                )

        if a.audit_info and shutil.which("bcftools"):
            cmd = ["bcftools", "query", "-r", f"{c}:{s}-{e}", "-f", "%CHROM\t%POS\t%INFO/AF\t%INFO/INFO\n", a.vcf]
            txt = subprocess.check_output(cmd, text=True)
            df_i = pd.read_csv(io.StringIO(txt), sep="\t", header=None, names=["chr", "pos", "af", "info"])
            df_i["af"] = pd.to_numeric(df_i["af"], errors="coerce")
            df_i["info"] = pd.to_numeric(df_i["info"], errors="coerce")
            df_i["maf"] = np.minimum(df_i["af"], 1.0 - df_i["af"])
            info_dfs.append(df_i)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(f"{a.out}/phased_vs_unphased.csv", index=False)
    print(f"Saved {len(out_df)} gene-trait phased comparisons to {a.out}/phased_vs_unphased.csv", flush=True)

    if info_dfs:
        ia = pd.concat(info_dfs, ignore_index=True)
        ia = ia[ia["maf"] >= 0.05].copy()
        ia["maf_bin"] = pd.cut(
            ia["maf"], bins=[0.05, 0.10, 0.20, 0.30, 0.501],
            labels=["0.05-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.50"],
        )
        isum = ia.groupby("maf_bin", observed=False)["info"].agg(
            n="count", mean="mean", median="median",
            pct_ge_08=lambda x: (x >= 0.8).mean(), pct_ge_09=lambda x: (x >= 0.9).mean(),
        )
        isum.to_csv(f"{a.out}/imputation_info_by_maf.csv")
        print("\n=== Imputation INFO by MAF bin ===")
        print(isum.round(4).to_string())


if __name__ == "__main__":
    main()
