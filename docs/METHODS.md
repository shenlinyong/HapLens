# Statistical methods (HapLens v1.0.0)

## 1. Null mixed model and whitening

For a standardised trait $y$, fixed covariates $X$ (`--covar`, `--cat-covar`), genomic relationship matrix $K$ (full or chromosome-specific LOCO GRM via `--loco-rel`), and optional group random effects $Z_r$ (e.g., dam effect $D$),

$$ y = X\beta + g + \sum_r u_r + e, \quad g \sim N(0, \sigma_g^2 K),\; u_r \sim N(0, \sigma_r^2 Z_r Z_r'),\; e \sim N(0, \sigma_e^2 I). $$

Variance components are estimated once under the null by REML (L-BFGS-B over log-variances). Data are whitened by $W = L^{-1}$ (Cholesky $V = LL'$) or by the order-invariant symmetric square root $W = V^{-1/2}$ (`--whiten sym`), giving $\tilde y = Wy$ and $\tilde X = WX$.

## 2. Window layer: omnibus mixed-model test (HapLens-O)

Adapting the ACAT-O omnibus principle (Liu et al., 2019; Liu and Xie, 2020) to family mixed models, each window (fixed-size or sliding via `make-windows --step`) is evaluated deterministically with **all** common SNPs (`--max-snp 0`) via three complementary components:

1. **Relatedness-kernel score test ($P_{\mathrm{kern}}$):** $Q = \lVert B'\tilde r\rVert^2 / \hat\sigma^2$ where $B = M\tilde Z$ and $M = I - \tilde X(\tilde X'\tilde X)^{-1}\tilde X'$. Nonzero eigenvalues $\lambda_k$ are computed from the smaller Gram matrix ($B'B$ if $m \le n$, else $BB'$). Upper-tail $P$ values use the Lugannani–Rice saddlepoint approximation (Kuonen, 1999).
2. **Within-window ACAT-V ($P_{\mathrm{ACAT\text{-}V}}$):** single-SNP GLS $F$-test $P$ values across all SNPs in the window are combined by ACAT, retaining power when a single major variant drives the signal.
3. **Haplotype-class $F$-test ($P_{\mathrm{cls}}$):** GLS $F$-test of unphased multilocus genotype combinations (diplotypes) across up to four LD-pruned tag SNPs.

These components are combined via ACAT without assuming independence.

## 3. Gene layer: phased haplotypes and Bi-LBC specificity screening

- **Phased haplotype encoding (`phased-genes`):** reads phased VCF/BCF (`|`) genotypes and compares additive phased haplotype dosages, phased diplotypes, unphased 0/1/2 combinations (Qi et al., 2025), and a 1-df coding kernel.
- **Bidirectional Local Background-Conditioned screening (`lbc-test`):** complements regional conditional gene-analysis approaches (such as summary-statistic conditional models) at the individual-level family mixed-model scale. For each gene, flanking non-coding SNPs within $\pm W$ bp outside a buffer around the gene body are summarized by orthonormal local PCs $U_{\mathrm{bg}}$ (capturing `--bg-var` of flanking variance, capped at `--max-bg-pcs`):
  - **Forward LBC ($P_{\mathrm{LBC}}$):** coding kernel score test after projecting out $[\tilde X, U_{\mathrm{bg}}]$;
  - **Reverse LBC ($P_{\mathrm{rev}}$):** flanking background kernel score test after projecting out $[\tilde X, U_{\mathrm{code}}]$;
  - **Collinearity diagnostics:** `max_r2_code_bg` and `bg_r2_explained` explicitly quantify coding–background collinearity;
  - **Matched neutral controls:** $B$ MAF-matched flanking neutral draws (`--n-ctrl 100` default) reporting median, 5th percentile, and empirical specificity $P$;
  - **Multiple testing:** Benjamini–Hochberg $q$-values (`q_code`, `q_lbc`) are computed within each trait.

**Scope and boundaries:** Bi-LBC evaluates statistical distinguishability between coding variants and their flanking genetic background. Attenuation of $P_{\mathrm{LBC}}$ under high coding–flanking LD indicates statistical indistinguishability (`collinear_indistinguishable`), not proof that the causal variant is non-coding.

## 4. Stratified permutation calibration

In `test-windows`, whitened null residuals $\tilde r$ are permuted within strata (e.g., sire) and **re-projected orthogonal to $\tilde X$** with exact per-permutation residual variance $\hat\sigma_p^2$. Setting `--perm 10` checks bulk calibration; evaluating extreme-tail false-positive rates ($P < 10^{-4}$, $P < 10^{-5}$, and Bonferroni $0.05/N$) requires `--perm 100` or greater. Because windows within a single permutation share the permuted phenotype and genomic LD, per-permutation min–max false-positive rates are also reported.

## References

- Kuonen D (1999) *Biometrika* 86:929–935.
- Liu Y et al. (2019) *Am J Hum Genet* 104:410–421.
- Liu Y, Xie J (2020) *J Am Stat Assoc* 115:393–402.
- Qi G et al. (2025) *Nat Commun* 16:4734.
- Wu MC et al. (2011) *Am J Hum Genet* 89:82–93.
