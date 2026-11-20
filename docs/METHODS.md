# Statistical methods

## Null model

For a trait with phenotype $y$ (standardised), sex and batch covariates $X$, genomic relationship matrix $K$ and dam incidence matrix $D$,

$$ y = X\beta + g + d + e,\quad g\sim N(0,\sigma_g^2K),\; d\sim N(0,\sigma_d^2DD'),\; e\sim N(0,\sigma_e^2I). $$

The variance components are estimated by REML (L-BFGS-B over the log-variances). With $V=\sigma_g^2K+\sigma_d^2DD'+\sigma_e^2I=LL'$, the phenotype and covariates are whitened, $\tilde y=L^{-1}y$ and $\tilde X=L^{-1}X$, and all tests use the whitened data with the variance components fixed at the null estimates.

## Haplotype-class test

Up to `--max-tags` tag SNPs per window are chosen by greedy pruning ($r^2<$ `--r2`) in order of decreasing minor allele frequency. The haplotype class of an individual is the combination of its 0/1/2 allele dosages at the tag SNPs; classes with fewer than `--min-n` individuals are dropped and their members enter as a nuisance class. The test is a GLS $F$-test of the dummy-coded classes against the most frequent class. For genes, the classes are the combinations of dosages at the functional variants of the gene.

## Relatedness-kernel test

With standardised genotypes $Z$ of the common SNPs in a window, the score statistic is

$$ Q=\frac{\lVert\tilde Z'\tilde r\rVert^2}{\hat\sigma^2}, $$

where $\tilde r$ is the residual of $\tilde y$ on $\tilde X$. Under the null, $Q\sim\sum_k\lambda_k\chi^2_1$ with $\lambda_k$ the eigenvalues of $(M\tilde Z)'(M\tilde Z)$, $M=I-\tilde X(\tilde X'\tilde X)^{-1}\tilde X'$. P values come from a saddlepoint approximation (Kuonen, 1999). Windows with more than `--max-snp` SNPs use a random subset.

## ACAT

$$ T=\frac1k\sum_{i=1}^k\tan\{(0.5-p_i)\pi\},\qquad p_{\mathrm{ACAT}}=0.5-\arctan(T)/\pi, $$

with $k=2$ for the two window tests (Liu and Xie, 2020). `combine-windows` also reports Bonferroni, suggestive ($1/N$) and Benjamini–Hochberg counts.

## Matched control haplotypes

For every gene, `build-genes` draws SNPs outside all annotated classes within 20 kb of the gene (widened to 100 and 500 kb if needed), one per functional variant and matched on allele frequency (±0.02). Several draws are made and the one whose number of classes and retained individuals is closest to the real haplotype is kept. The control is tested with the same model.

## Calibration and power

The genomic inflation factor is the median chi-square divided by 0.455. Sire-stratified permutations shuffle haplotype labels (`test-genes`) or whitened residuals (`test-windows`) within sires. `power-windows` and `power-genes` add an additive effect with a given fraction of the phenotypic variance to the observed phenotype and repeat the tests with the null covariance fixed.

## Follow-up

`conditional` adds lead-SNP genotypes to the covariates. `local-pve` fits REML with the window kernel as a fourth variance component. `robustness` recomputes the kernel P value after excluding one sire at a time and in random sire split-halves. `check-gene` combines these analyses for one gene.

## References

Kuonen D (1999) Biometrika 86:929–935. Liu Y, Xie J (2020) J Am Stat Assoc 115:393–402. Wu MC et al. (2011) Am J Hum Genet 89:82–93.
