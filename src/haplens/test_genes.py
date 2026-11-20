"""Mixed-model association of haplotype classes (genes or windows) with traits."""

import argparse, os, multiprocessing as mp, numpy as np, pandas as pd
from scipy import stats
from haplens.model import reml_fit, whiten, scan


_CTX = {}


def _fwer_worker(seed):
    """One sire-stratified permutation of the haplotype labels; returns the minimum P and hit counts."""
    c = _CTX
    rng = np.random.default_rng(int(seed))
    perm = np.arange(c["n"])
    for idxs in c["strata"]:
        idxs = np.array(idxs)
        perm[idxs] = rng.permutation(idxs)
    ps = np.array([x["P"] for x in scan(c["yt"], c["Xt"], c["W"], c["L1"][:, perm], want_effects=False) if x])
    return ps.min(), int((ps < 0.001).sum()), int((ps < 0.05 / len(ps)).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="haplens test-genes",
        description="Mixed-model association of haplotype classes (genes or windows) with traits.",
    )
    ap.add_argument(
        "--no-sex-traits",
        default="",
        help="comma-separated traits analysed without the sex covariate (e.g. a trait measured at birth)",
    )
    ap.add_argument(
        "--fh", required=True, help="haplotype file written by build-genes or build-windows (.npz)"
    )
    ap.add_argument(
        "--pheno",
        required=True,
        help="phenotype table with columns IID, sex, batch, sire, dam and one column per trait",
    )
    ap.add_argument("--rel", required=True, help="prefix of the PLINK square GRM (.rel and .rel.id)")
    ap.add_argument("--traits", default=None, help="comma-separated trait names")
    ap.add_argument("--random", default="grm,dam")
    ap.add_argument("--out", default="res", help="output directory or file prefix")
    ap.add_argument("--perm", type=int, default=0, help="number of sire-stratified permutations")
    ap.add_argument("--fwer-perm", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--perm-genes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7, help="random seed")
    a = ap.parse_args(argv)
    no_sex = set(filter(None, a.no_sex_traits.split(",")))
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    z = np.load(a.fh, allow_pickle=True)
    genes = z["genes"]
    FH = z["fh"]
    CT = z["ctl"]
    iid_fh = z["iid"].astype(str)
    ph = pd.read_csv(a.pheno, dtype={"IID": str}).set_index("IID")
    rid = pd.read_csv(a.rel + ".rel.id", sep=r"\s+", header=None, dtype=str)[1].values
    R = np.loadtxt(a.rel + ".rel")
    rpos = {i: k for k, i in enumerate(rid)}
    fpos = {i: k for k, i in enumerate(iid_fh)}
    traits = (
        a.traits.split(",")
        if a.traits
        else [c for c in ph.columns if c not in ("sex", "batch", "sire", "dam")]
    )
    summ = []
    for t in traits:
        d = ph[ph[t].notna() & ph.sex.notna() & ph.index.isin(rid) & ph.index.isin(iid_fh)]
        ids = d.index.values
        n = len(ids)
        if n < 300:
            print("skip", t, n)
            continue
        y = d[t].values.astype(float)
        y = (y - y.mean()) / y.std()
        cov = [np.ones(n)]
        if t not in no_sex:
            cov.append(d.sex.values.astype(float))
        for b in sorted(d.batch.unique())[1:]:
            cov.append((d.batch.values == b).astype(float))
        X = np.column_stack(cov)
        K = R[np.ix_([rpos[i] for i in ids], [rpos[i] for i in ids])] + 0.001 * np.eye(n)
        mats = []
        names = []
        if "grm" in a.random:
            mats.append(K)
            names.append("grm")
        if "dam" in a.random:
            dm = d.dam.values
            mats.append((dm[:, None] == dm[None, :]).astype(float))
            names.append("dam")
        if "sire" in a.random:
            sr = d.sire.values
            mats.append((sr[:, None] == sr[None, :]).astype(float))
            names.append("sire")
        comp, V = reml_fit(y, X, mats)
        W = whiten(V)
        yt = W @ y
        Xt = W @ X
        sel = [fpos[i] for i in ids]
        L1 = FH[:, sel]
        L0 = CT[:, sel]
        r1 = scan(yt, Xt, W, L1)
        r0 = scan(yt, Xt, W, L0, want_effects=False)
        rows = []
        for g, x, c in zip(genes, r1, r0):
            if x is None:
                continue
            rows.append(
                dict(
                    gene_id=g,
                    df=x["df"],
                    F=x["F"],
                    P=x["P"],
                    pve=x["pve"],
                    ref=x["ref"],
                    eff=";".join((f"{k}:{b:.3f}:{s:.3f}" for k, b, s in x["eff"])),
                    P_ctl=c["P"] if c else np.nan,
                    pve_ctl=c["pve"] if c else np.nan,
                    df_ctl=c["df"] if c else np.nan,
                )
            )
        res = pd.DataFrame(rows)
        res.to_csv(f"{a.out}/{t}.csv", index=False)
        lam = lambda p: float(np.median(stats.chi2.isf(p, 1)) / 0.4549364)
        both = res.dropna(subset=["P_ctl"])
        wil = stats.wilcoxon(-np.log10(both.P), -np.log10(both.P_ctl)).pvalue if len(both) > 20 else np.nan
        h = dict(
            trait=t,
            n=n,
            **{f"var_{k}": float(v) for k, v in zip(names + ["e"], comp)},
            h2_grm=float(comp[0] / comp.sum()),
            genes=len(res),
            lam_fh=lam(res.P),
            lam_ctl=lam(both.P_ctl) if len(both) else np.nan,
            n_p001_fh=int((res.P < 0.001).sum()),
            n_p001_ctl=int((both.P_ctl < 0.001).sum()),
            n_bonf_fh=int((res.P < 0.05 / len(res)).sum()),
            n_bonf_ctl=int((both.P_ctl < 0.05 / len(both)).sum()) if len(both) else 0,
            wilcoxon_p=wil,
        )
        if a.perm:
            sub = rng.choice(len(genes), size=min(a.perm_genes, len(genes)), replace=False)
            strata = pd.Series(np.arange(n)).groupby(d.sire.values).apply(list)
            pp = []
            for _ in range(a.perm):
                perm = np.arange(n)
                for idxs in strata:
                    idxs = np.array(idxs)
                    perm[idxs] = rng.permutation(idxs)
                for x in scan(yt, Xt, W, L1[sub][:, perm], want_effects=False):
                    if x:
                        pp.append(x["P"])
            pp = np.array(pp)
            h.update(
                perm_lambda=lam(pp),
                perm_p05=float((pp < 0.05).mean()),
                perm_p001=float((pp < 0.001).mean()),
                perm_n=len(pp),
            )
        if a.fwer_perm:
            strata = pd.Series(np.arange(n)).groupby(d.sire.values).apply(list)
            seeds = rng.integers(0, 2**31 - 1, a.fwer_perm)

            _CTX.update(n=n, strata=strata, yt=yt, Xt=Xt, W=W, L1=L1)

            with mp.get_context("fork").Pool(a.jobs) as pool:
                pr = pool.map(_fwer_worker, seeds)
            pr = np.array(pr)
            pd.DataFrame(pr, columns=["min_P", "n_p001", "n_bonf"]).to_csv(
                f"{a.out}/{t}_fwer_perm.csv", index=False
            )
            h.update(
                fwer_thr_05=float(np.quantile(pr[:, 0], 0.05)),
                fwer_perms=len(pr),
                perm_mean_n_p001=float(pr[:, 1].mean()),
            )
        summ.append(h)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in h.items()}, flush=True)
    pd.DataFrame(summ).to_csv(f"{a.out}/summary_{'-'.join(traits)[:90]}.csv", index=False)


if __name__ == "__main__":
    main()
