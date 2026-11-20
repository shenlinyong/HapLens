"""Resampling, conditional and control analyses for one gene-level signal."""

import argparse, os, sys, numpy as np, pandas as pd
from scipy import stats
from haplens.model import reml_fit, whiten, scan


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens check-gene",
        description="Resampling, conditional and control analyses for one gene-level signal.",
    )
    ap.add_argument(
        "--no-sex-traits",
        default="",
        help="comma-separated traits analysed without the sex covariate (e.g. a trait measured at birth)",
    )
    ap.add_argument(
        "--fh", required=True, help="haplotype file written by build-genes or build-windows (.npz)"
    )
    ap.add_argument("--genes-tsv", required=True)
    ap.add_argument(
        "--pheno",
        required=True,
        help="phenotype table with columns IID, sex, batch, sire, dam and one column per trait",
    )
    ap.add_argument("--rel", required=True, help="prefix of the PLINK square GRM (.rel and .rel.id)")
    ap.add_argument("--bfile", required=True, help="PLINK binary genotype prefix (.bed/.bim/.fam)")
    ap.add_argument("--gene-id", required=True)
    ap.add_argument("--gene", default="")
    ap.add_argument("--focus", required=True)
    ap.add_argument("--traits", default=None, help="comma-separated trait names")
    ap.add_argument("--lead-snps", default="")
    ap.add_argument("--locus-csv", default=None)
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--splits", type=int, default=100, help="number of random sire split-halves")
    ap.add_argument("--seed", type=int, default=11, help="random seed")
    ap.add_argument("--out", default="gene_check", help="output directory or file prefix")
    a = ap.parse_args(argv)
    no_sex = set(filter(None, a.no_sex_traits.split(",")))
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    z = np.load(a.fh, allow_pickle=True)
    genes = list(z["genes"])
    FH = z["fh"]
    CT = z["ctl"]
    iid_fh = z["iid"].astype(str)
    fpos = {i: k for k, i in enumerate(iid_fh)}
    gi = genes.index(a.gene_id)
    GM = pd.read_csv(a.genes_tsv, sep="\t", dtype={"chr": str}).set_index("gene_id")
    gchr = GM.loc[a.gene_id, "chr"]
    label = a.gene or GM.loc[a.gene_id, "gene_name"]
    ph = pd.read_csv(a.pheno, dtype={"IID": str}).set_index("IID")
    rid = pd.read_csv(a.rel + ".rel.id", sep=r"\s+", header=None, dtype=str)[1].values
    R = np.loadtxt(a.rel + ".rel")
    rpos = {i: k for k, i in enumerate(rid)}
    traits = (
        a.traits.split(",")
        if a.traits
        else [c for c in ph.columns if c not in ("sex", "batch", "sire", "dam")]
    )
    if a.focus not in traits:
        traits.append(a.focus)
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
    keepb = fam[1].isin(fpos).values
    iid_b = fam[1].values[keepb]
    bpos = {i: k for k, i in enumerate(iid_b)}
    bed = np.memmap(a.bfile + ".bed", dtype=np.uint8, mode="r", offset=3, shape=(len(bim), bpl))
    SH = np.array([0, 2, 4, 6], dtype=np.uint8)
    LUT = np.array([2, -1, 1, 0], dtype=np.int8)

    def snp_by_pos(c, p):
        i = np.nonzero((bim.chr.values == c) & (bim.pos.values == int(p)))[0]
        if len(i) == 0:
            return None
        g = LUT[(np.asarray(bed[i[:1]])[:, :, None] >> SH & 3).reshape(1, -1)[:, :n_all][:, keepb]][0].astype(
            float
        )
        g[g < 0] = np.nan
        g[np.isnan(g)] = np.nanmean(g)
        return g

    vpos = [int(v.split(":")[0]) for v in str(GM.loc[a.gene_id, "variants"]).split(";")]
    vinfo = [":".join(v.split(":")[1:]) for v in str(GM.loc[a.gene_id, "variants"]).split(";")]
    VG = {f"{gchr}:{p}": snp_by_pos(gchr, p) for p in vpos}
    LG = {}
    for s in [x for x in a.lead_snps.split(",") if x]:
        c, p = s.split(":")
        g = snp_by_pos(c, p)
        if g is not None:
            LG[s] = g

    def align(ids, g):
        return g[[bpos[i] for i in ids]]

    class T:
        pass

    S = {}
    for t in traits:
        d = ph[ph[t].notna() & ph.sex.notna() & ph.index.isin(rid) & ph.index.isin(iid_fh)]
        o = T()
        o.ids = d.index.values
        n = len(o.ids)
        o.n = n
        o.sire = d.sire.values
        y = d[t].values.astype(float)
        o.y = (y - y.mean()) / y.std()
        cov = (
            [np.ones(n)]
            + ([d.sex.values.astype(float)] if t not in no_sex else [])
            + [(d.batch.values == b).astype(float) for b in sorted(d.batch.unique())[1:]]
        )
        o.X = np.column_stack(cov)
        K = R[np.ix_([rpos[i] for i in o.ids], [rpos[i] for i in o.ids])] + 0.001 * np.eye(n)
        dm = d.dam.values
        comp, o.V = reml_fit(o.y, o.X, [K, (dm[:, None] == dm[None, :]).astype(float)])
        o.W = whiten(o.V)
        o.yt = o.W @ o.y
        o.Xt = o.W @ o.X
        o.sel = [fpos[i] for i in o.ids]
        S[t] = o

    def run(o, L, idx=None, Xextra=None):
        """test labels L (G,n_o) on trait object o, optionally on the individual subset idx (V submatrix, comps fixed)"""
        if idx is None:
            W, yt, Xt = (o.W, o.yt, o.Xt)
            X = o.X
        else:
            W = whiten(o.V[np.ix_(idx, idx)])
            X = o.X[idx]
            yt = W @ o.y[idx]
            Xt = W @ X
            L = L[:, idx]
        if Xextra is not None:
            Xe = Xextra if idx is None else Xextra[idx]
            Xt = np.hstack([Xt, W @ Xe])
        return scan(yt, Xt, W, L, want_effects=True)

    log = open(f"{a.out}/summary.txt", "w")

    def say(*x):
        s = " ".join((str(i) for i in x))
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    obs = []
    for t in traits:
        r = run(S[t], FH[[gi]][:, S[t].sel])[0]
        if r:
            obs.append(dict(trait=t, n=S[t].n, df=r["df"], P=r["P"], pve=r["pve"]))
    obs = pd.DataFrame(obs)
    obs.to_csv(f"{a.out}/observed_all_traits.csv", index=False)
    say(f"== {label} ({a.gene_id}) observed, all traits")
    say(obs.sort_values("P").head(6).to_string(index=False))
    o = S[a.focus]
    Lg = FH[[gi]][:, o.sel]
    r_full = run(o, Lg)[0]
    say(f"focus {a.focus}: P={r_full['P']:.3g} df={r_full['df']} pve={r_full['pve']:.4f}")
    cls = pd.DataFrame([dict(fh_class=k, beta=b, se=s, z=b / s) for k, b, s in r_full["eff"]])
    cnt = pd.Series(Lg[0][Lg[0] >= 0]).value_counts()
    cls["n"] = cls.fh_class.map(cnt)
    cls.to_csv(f"{a.out}/focus_class_effects.csv", index=False)
    say("reference class n =", int(cnt.get(r_full["ref"], 0)), "; classes:", len(cnt))
    rows = []
    for s_ in np.unique(o.sire):
        idx = np.nonzero(o.sire != s_)[0]
        r = run(o, Lg, idx)[0]
        if r:
            rows.append(dict(dropped_sire=s_, n_dropped=int((o.sire == s_).sum()), P=r["P"], df=r["df"]))
    loso = pd.DataFrame(rows)
    loso.to_csv(f"{a.out}/loso.csv", index=False)
    say(
        f"LOSO: P range {loso.P.min():.2g} – {loso.P.max():.2g}; worst sire {loso.loc[loso.P.idxmax(), 'dropped_sire']}; all < 0.001: {(loso.P < 0.001).all()}"
    )
    sires = np.unique(o.sire)
    rows = []
    for k in range(a.splits):
        perm = rng.permutation(sires)
        A = np.isin(o.sire, perm[: len(perm) // 2])
        ra = run(o, Lg, np.nonzero(A)[0])[0]
        rb = run(o, Lg, np.nonzero(~A)[0])[0]
        if ra and rb:
            rows.append(dict(split=k, P_A=ra["P"], P_B=rb["P"], df_A=ra["df"], df_B=rb["df"]))
    sp = pd.DataFrame(rows)
    sp.to_csv(f"{a.out}/split_half.csv", index=False)
    say(
        f"split-half ({len(sp)} splits): median P {sp[['P_A', 'P_B']].stack().median():.2g}; both P<0.05 in {100 * ((sp.P_A < 0.05) & (sp.P_B < 0.05)).mean():.0f}% ; both P<0.01 in {100 * ((sp.P_A < 0.01) & (sp.P_B < 0.01)).mean():.0f}%"
    )
    rows = []
    for (key, g), inf in zip(VG.items(), vinfo):
        if g is None:
            continue
        ga = align(o.ids, g)
        Xe = ga[:, None]
        Xt = np.hstack([o.Xt, o.W @ Xe])
        c, *_ = np.linalg.lstsq(Xt, o.yt, rcond=None)
        res = o.yt - Xt @ c
        dfd = o.n - Xt.shape[1]
        s2 = float(res @ res) / dfd
        se = np.sqrt(s2 * np.linalg.inv(Xt.T @ Xt)[-1, -1])
        b = c[-1]
        af = ga.mean() / 2
        rows.append(
            dict(
                variant=key,
                annotation=inf,
                maf=min(af, 1 - af),
                beta=b,
                se=se,
                P=2 * stats.t.sf(abs(b / se), dfd),
            )
        )
    vt = pd.DataFrame(rows)
    vt.to_csv(f"{a.out}/variant_tests.csv", index=False)
    say("single-variant tests in the gene (additive):")
    say(vt.to_string(index=False))
    names = list(VG.keys()) + list(LG.keys())
    G = np.array(
        [
            align(o.ids, VG[k]) if k in VG and VG[k] is not None else align(o.ids, LG[k])
            for k in names
            if k in VG and VG[k] is not None or k in LG
        ]
    )
    nm2 = [k for k in names if k in VG and VG[k] is not None or k in LG]
    pd.DataFrame(np.corrcoef(G) ** 2, index=nm2, columns=nm2).to_csv(f"{a.out}/ld_r2.csv")
    say(
        "max r2 between gene variants and lead SNPs:",
        {
            k: round(
                float(
                    max(
                        (
                            np.corrcoef(align(o.ids, VG[v]), align(o.ids, LG[k]))[0, 1] ** 2
                            for v in VG
                            if VG[v] is not None
                        )
                    )
                ),
                3,
            )
            for k in LG
        },
    )
    cond = [dict(model="unconditional", P=r_full["P"], df=r_full["df"])]
    if LG:
        Xe = np.column_stack([align(o.ids, g) for g in LG.values()])
        r = run(o, Lg, None, Xe)[0]
        cond.append(dict(model="| lead SNPs " + "+".join(LG), P=r["P"], df=r["df"]))
    if len(vt):
        best = vt.loc[vt.P.idxmin(), "variant"]
        r = run(o, Lg, None, align(o.ids, VG[best])[:, None])[0]
        cond.append(dict(model=f"| best variant {best}", P=r["P"], df=r["df"]))
        burden = np.sum([align(o.ids, g) for g in VG.values() if g is not None], 0)
        r = run(o, Lg, None, burden[:, None])[0]
        cond.append(dict(model="| additive burden of all variants", P=r["P"], df=r["df"]))
    Cg = CT[[gi]][:, o.sel][0]
    if (Cg >= 0).any():
        cl_ = [c for c in np.unique(Cg[Cg >= 0])]
        Xc = (
            np.column_stack([(Cg == c).astype(float) for c in cl_[1:]] + [(Cg < 0).astype(float)])
            if len(cl_) > 1
            else None
        )
        if Xc is not None and Xc.shape[1] > 0:
            r = run(o, Lg, None, Xc)[0]
            cond.append(dict(model="| matched neutral control haplotype", P=r["P"], df=r["df"]))
            rc2 = run(
                o,
                CT[[gi]][:, o.sel],
                None,
                np.column_stack(
                    [(Lg[0] == c).astype(float) for c in np.unique(Lg[0][Lg[0] >= 0])[1:]]
                    + [(Lg[0] < 0).astype(float)]
                ),
            )[0]
            cond.append(dict(model=f"control haplotype | {label} FH", P=rc2["P"], df=rc2["df"]))
    pd.DataFrame(cond).to_csv(f"{a.out}/conditional.csv", index=False)
    say("conditional FH tests:")
    say(pd.DataFrame(cond).to_string(index=False))
    res = run(o, FH[:, o.sel])
    dfs = np.array([x["df"] if x else -1 for x in res])
    Ps = np.array([x["P"] if x else np.nan for x in res])
    m = (np.abs(dfs - r_full["df"]) <= 2) & ~np.isnan(Ps)
    say(
        f"genome-wide {a.focus}: {m.sum()} genes with df within ±2 of {r_full['df']}; rank of {label}: {int((Ps[m] <= r_full['P']).sum())} (empirical P = {(Ps[m] <= r_full['P']).sum() / m.sum():.2g}); lambda of this df-stratum = {np.median(stats.chi2.isf(Ps[m], 1)) / 0.4549:.2f}"
    )
    rc = run(o, CT[[gi]][:, o.sel], None)[0]
    say(
        f"matched neutral control haplotype of {label}: P={(rc['P'] if rc else float('nan')):.3g} (df {(rc['df'] if rc else '-')})"
    )
    pd.DataFrame(dict(gene_id=np.array(genes), df=dfs, P=Ps)).to_csv(
        f"{a.out}/genomewide_focus_trait.csv", index=False
    )
    if a.perms == 0:
        say("done (no permutations)")
        sys.exit(0)
    if a.locus_csv:
        lg = pd.read_csv(a.locus_csv)
        lg = lg[lg.tier == "A"].gene_id.unique()
        li = [genes.index(g) for g in lg if g in genes]
    else:
        li = [gi]
    if gi not in li:
        li.append(gi)
    pos_g = li.index(gi)
    obs_min = min((x["P"] for t in traits for x in run(S[t], FH[li][:, S[t].sel]) if x))
    obs_gene = min((x["P"] for t in traits for x in [run(S[t], FH[[gi]][:, S[t].sel])[0]] if x))
    all_sire = ph.loc[iid_fh, "sire"].values if set(iid_fh) <= set(ph.index) else None
    strata = pd.Series(np.arange(len(iid_fh))).groupby(all_sire).apply(list)
    mins, mins_g = ([], [])
    for b in range(a.perms):
        perm = np.arange(len(iid_fh))
        for idxs in strata:
            idxs = np.array(idxs)
            perm[idxs] = rng.permutation(idxs)
        Lp = FH[li][:, perm]
        pm, pg = (1.0, 1.0)
        for t in traits:
            r = scan(S[t].yt, S[t].Xt, S[t].W, Lp[:, S[t].sel], want_effects=False)
            ps = [x["P"] for x in r if x]
            if ps:
                pm = min(pm, min(ps))
            if r[pos_g]:
                pg = min(pg, r[pos_g]["P"])
        mins.append(pm)
        mins_g.append(pg)
        if (b + 1) % 200 == 0:
            say(f"  perm {b + 1}/{a.perms}", flush=True) if False else None
    mins = np.array(mins)
    mins_g = np.array(mins_g)
    say(
        f"FWER permutation ({a.perms} perms; {len(li)} locus genes x {len(traits)} traits): observed min P {obs_min:.3g}; FWER P = {(1 + (mins <= obs_min).sum()) / (1 + a.perms):.4g}; perm-null 5% quantile of min P = {np.quantile(mins, 0.05):.3g}"
    )
    say(
        f"gene-specific (best of {len(traits)} traits for {label}): observed {obs_gene:.3g}; P = {(1 + (mins_g <= obs_gene).sum()) / (1 + a.perms):.4g}"
    )
    pd.DataFrame(dict(minP_locus=mins, minP_gene=mins_g)).to_csv(f"{a.out}/perm_minP.csv", index=False)
    say("done")


if __name__ == "__main__":
    main()
