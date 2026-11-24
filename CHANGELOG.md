# Changelog

## 1.0.0 (2026-11-23)

- **HapLens-O omnibus window test (`test-windows`, `combine-windows`):** combines an exact all-SNP
  relatedness-kernel score test (dual Gram eigendecomposition, `--max-snp 0` default, cancellation-free
  Lugannani–Rice saddlepoint tail evaluation), within-window ACAT-V (exact `scipy.special.fdtrc`), and
  tag-SNP haplotype-class tests under one calibrated mixed model.
- **Bidirectional Local Background-Conditioned (Bi-LBC) specificity screening (`lbc-test`, `bench-lbc`):**
  pairs forward conditioning on flanking non-coding local PCs with reverse conditioning, collinearity
  diagnostics (`max_r2_code_bg`, `bg_r2_explained`), within-trait BH `q_code`/`q_lbc`, descriptive
  non-causal status labels, default `--n-ctrl 100` matched neutral draws (`P_ctrl_q05`), and LD-stratified
  hitchhiking benchmarks with 95% Wilson CIs.
- **True phased haplotype module (`phased-genes`):** reads phased VCF/BCF (`|` genotypes) to compare
  phased additive haplotype dosages, phased diplotypes, unphased combinations, and 1-df coding kernels,
  plus MAF-stratified imputation INFO auditing.
- **Calibration, power benchmarks & engineering:** orthogonally re-projected sire-stratified permutation
  with exact per-permutation variance (`float64` output + extreme-tail statistics to Bonferroni threshold),
  order-invariant symmetric whitening (`--whiten sym`), LOCO-GRM support (`--loco-rel`), flexible
  covariate/random-effect CLI (`--covar`, `--cat-covar`, `--random`, `--strata`), sliding windows
  (`make-windows --step`), multi-architecture 6-method power benchmark (`power-windows`, separating
  concordant and opposing multi-causal variants plus 2-SNP cis-epistasis with 95% Wilson CIs, runtime,
  and peak RAM logging), and unified `haplens run`.

## 0.1.0 (2026-11-20)

- Initial command-line interface `haplens` with 18 commands and simulated population tests.
