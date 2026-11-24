# HapLens

**Multi-scale haplotype association and local background-conditioned specificity screening for family-structured populations.**

HapLens is a Python package designed for genetic association analysis in family-structured livestock and pedigree populations at both window and gene scales under one mixed model:

1. **Family mixed-model calibration:** fits REML with a genomic relationship matrix (full or chromosome-specific `--loco-rel`), flexible covariates (`--covar`, `--cat-covar`), and pedigree/common-environment random effects (`--random grm,dam`), supporting Cholesky or order-invariant symmetric whitening (`--whiten sym`) and orthogonally re-projected stratified permutations. Note: `--perm 10` (default) provides a fast check of bulk calibration; evaluating extreme-tail false-positive rates ($P < 10^{-4}$ to Bonferroni) requires `--perm 100` or more.
2. **Omnibus window testing (`test-windows`, `combine-windows`):** adapts the ACAT-O omnibus strategy (Liu et al., 2019; Liu and Xie, 2020) to family mixed models, combining an all-SNP relatedness-kernel score test (exact dual eigendecomposition, `--max-snp 0`), within-window ACAT-V, and a tag-SNP haplotype-class test without assuming independence.
3. **Bidirectional Local Background-Conditioned specificity screening (`lbc-test`, `bench-lbc`):** evaluates whether a gene-level coding association persists after conditioning on top local PCs of flanking non-coding variants (forward $P_{\mathrm{LBC}}$) and tests the reverse direction ($P_{\mathrm{rev}}$), alongside multi-draw MAF-matched neutral controls (`--n-ctrl 100` default) and collinearity metrics (`max_r2_code_bg`, `bg_r2_explained`). Bi-LBC is an association-specificity screening tool, not a causal-variant proof: when coding and flanking variants are tightly collinear, signals are explicitly flagged as statistically indistinguishable (`collinear_indistinguishable`).
4. **Phased haplotype comparison (`phased-genes`):** reads phased VCF/BCF (`|` genotypes) to compare phased additive haplotype dosages, phased diplotypes, unphased multilocus genotype combinations, and 1-df coding kernels, with MAF-stratified imputation `INFO` auditing.

## Installation & Quick Start

```bash
pip install -e .
haplens simulate --out example/data
haplens run --bfile example/data/ex --pheno example/data/pheno.csv --rel example/data/grm \
  --traits T0,T1 --chroms 1 --ann example/data/ann_variants.tsv --genes example/data/ann_genes.tsv \
  --out example/run_out
```

`examples/run_example.sh` runs a small end-to-end demonstration on 480 simulated individuals (~15 s) for functional verification; full-scale calibration and power benchmarks are evaluated via `test-windows --perm 100`, `power-windows`, and `bench-lbc` on real genotypes.

## Commands

| Command | Purpose |
|---|---|
| `run` | one-command end-to-end workflow (window omnibus + optional gene Bi-LBC) |
| `simulate` | simulate a small structured population (functional examples and unit tests) |
| `make-windows` | generate fixed-size or sliding (`--step`) windows from a `.bim` file |
| `annotate` | classify SNPs by predicted functional consequence from GTF + FASTA |
| `build-genes` | construct gene-level functional diplotypes and matched neutral controls |
| `test-genes` | mixed-model GLS $F$-test of haplotype/diplotype classes |
| `lbc-test` | Bidirectional Local Background-Conditioned (Bi-LBC) coding specificity screening |
| `bench-lbc` | LD-stratified benchmark of Bi-LBC under true coding vs flanking non-coding QTLs |
| `phased-genes` | phased haplotype dosage / diplotype / coding-kernel comparison from phased VCF/BCF |
| `build-windows` | tag-SNP haplotype classes for windows |
| `test-windows` | all-SNP relatedness-kernel + ACAT-V + omnibus window scan |
| `combine-windows` | combine kernel, ACAT-V and class tests into omnibus and multi-trait ACAT |
| `snp-scan` | single-SNP GLS scan under the same null mixed model |
| `conditional` | window kernel test conditional on lead SNPs |
| `local-pve` | local variance explained by a window (REML) |
| `robustness` | leave-one-sire-out and sire split-half stability of windows |
| `power-windows` | multi-architecture power benchmark (1-causal, concordant/opposing 3-causal, 2-SNP epistasis) |
| `power-genes` | gene-level haplotype power by number of classes |
| `null-compare` | compare random-effect structures by genomic inflation |
| `check-gene` | resampling, conditional and control dissection of a single gene |
| `pc-check` | relate GRM principal components to pedigree and design factors |
| `export-tables` | export minimum SNP $P$ per window, GRM PCs and regional gene tables |

Run `haplens <command> --help` for full options. See [docs/METHODS.md](docs/METHODS.md) and [docs/FORMATS.md](docs/FORMATS.md).

## Tests

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
```

## Licence and contact

MIT licence. Linyong Shen, College of Animal Science and Technology, Northwest A&F University, Yangling, Shaanxi, China (`shenlinyong@nwafu.edu.cn`).
