# HapLens

**Multi-scale haplotype association for family-structured populations.**

HapLens tests haplotype units at two scales, fixed-size windows and genes, under one mixed model. The null model carries the genomic relationship matrix (GRM) and a dam effect, and it is checked against sire-stratified permutations before any association is read. At the window scale, a haplotype-class test and a regional relatedness-kernel test are combined by the aggregated Cauchy association test (ACAT). At the gene scale, functional haplotypes are built from predicted protein-altering variants, and every gene is paired with a matched neutral control haplotype. Single-SNP scans, conditional tests, local variance estimates, leave-one-sire-out checks and power simulations use the same null model, so the layers can be compared on one scale.

HapLens was developed for the F1 population of Lueyang black-boned chickens described in the manuscript cited below. It needs PLINK genotypes, a phenotype table with pedigree columns and a GRM. It does not need phased data.

```bash
pip install -e .
haplens simulate --out example/data
haplens build-genes --bfile example/data/ex --ann example/data/ann_variants.tsv \
  --genes example/data/ann_genes.tsv --pheno example/data/pheno.csv --tier A --out example/genesA
haplens test-genes --fh example/genesA_fh.npz --pheno example/data/pheno.csv \
  --rel example/data/grm --out example/res_genes
```

`examples/run_example.sh` runs the window and gene layers end to end on the simulated population (about 10 s).

## What HapLens changes

**1. Family structure.**

*Problem:* in an F1 cross, sires and dams share genotypes and environments, so a model with sire and dam effects only gives inflated tests. In the Lueyang population the inflation factor of a gene-level test was 3.4–5.7 without the GRM.

*HapLens:* every trait is fitted once with sex and batch as fixed effects and the GRM and a dam effect as random effects (REML). The variance components are fixed and all tests are generalised least squares on the whitened data. `test-genes` and `test-windows` can write sire-stratified permutations of the haplotype labels or residuals (`--perm`), which give the empirical inflation factor and false-positive rate.

**2. Degrees of freedom of haplotype classes.**

*Problem:* a test of haplotype classes spends one degree of freedom per class, and many classes are rare. Power falls as the number of classes grows.

*HapLens:* each 50-kb window is tested by a regional relatedness-kernel score test of all SNPs (one variance component, saddlepoint P value) and by an *F*-test of up to four tag-SNP classes. The two P values are combined by ACAT, and ACAT can be applied again across traits. `power-windows` and `power-genes` measure the power of each test by adding a simulated effect to the observed phenotype.

**3. Gene-level signals that reflect the region.**

*Problem:* a gene-level haplotype can be associated because it tags the surrounding region and not because of its coding variants.

*HapLens:* `build-genes` draws a neutral control haplotype for every gene from nearby SNPs outside all annotated classes, matched on allele frequency, number of classes and number of retained individuals. `test-genes` reports both. `check-gene` repeats a signal with leave-one-sire-out, sire split-halves, family-wise permutations, single-variant tests and tests conditional on lead SNPs and on the control haplotype.

**4. Dependence of window P values on SNP sampling.**

*Problem:* with hundreds of SNPs per window, a random subset (`--max-snp`) changes the kernel P value by up to an order of magnitude for some windows.

*HapLens:* `--max-snp` and `--seed` make the subset explicit. Setting `--max-snp` above the number of SNPs in a window uses all SNPs (`test-windows`, `conditional`, `robustness`), and rerunning with several seeds shows the spread of the P value.

## Commands

| Command | Purpose |
|---|---|
| `simulate` | simulate a small structured population (examples, tests) |
| `make-windows` | write fixed-size windows from a `.bim` file |
| `annotate` | classify SNPs by predicted consequence from a GTF and a genome FASTA |
| `build-genes` | gene-level functional haplotypes and matched control haplotypes |
| `build-windows` | haplotype classes of tag SNPs for windows |
| `test-genes` | mixed-model test of haplotype classes (genes or windows) |
| `test-windows` | regional relatedness-kernel test of windows |
| `combine-windows` | ACAT combination of the two window tests and a window report |
| `snp-scan` | single-SNP scan under the null mixed model |
| `conditional` | window kernel test conditional on lead SNPs |
| `local-pve` | variance explained by a window |
| `robustness` | leave-one-sire-out and sire split-half stability of windows |
| `power-windows`, `power-genes` | power of the window and gene tests by simulation |
| `null-compare` | compare random-effect structures by the inflation of the gene-level test |
| `check-gene` | resampling, conditional and control analyses of one gene-level signal |
| `pc-check` | relate GRM principal components to the pedigree |
| `export-tables` | minimum SNP P per window, GRM principal components, gene results of regions |

Run `haplens <command> --help` for the options. Formats are described in [docs/FORMATS.md](docs/FORMATS.md) and the statistics in [docs/METHODS.md](docs/METHODS.md).

## Input

- **Genotypes:** PLINK 1 binary files (`.bed`, `.bim`, `.fam`), biallelic SNPs. The `.fam` individual IDs must match the `IID` column of the phenotype table.
- **Phenotypes:** a CSV with `IID`, `sex`, `batch`, `sire`, `dam` and one column per trait. Missing values are empty. Individuals without sex, sire or dam should be removed beforehand.
- **GRM:** the square matrix written by `plink --make-rel square` (`.rel` and `.rel.id`) from a random set of common autosomal SNPs.
- **Annotation (gene layer):** a GTF file and the genome FASTA, converted by `annotate` into `ann_variants.tsv` and `ann_genes.tsv`.
- **Windows (window layer):** a table written by `make-windows`.

## Analysis of the Lueyang black-boned chicken population

The commands below are those used for the manuscript. The hatch-weight trait `BW0` was analysed without the sex covariate (`--no-sex-traits BW0`). Traits were run in groups of three in parallel jobs.

```bash
# GRM (PLINK 1.9)
plink --bfile wuji_bed --keep keep_ids.txt --chr 1-38 --maf 0.05 --geno 0.05 \
  --thin-count 200000 --seed 1 --make-rel square --out grm

# gene layer, tiers A and B
haplens annotate --bim wuji_bed.bim --gtf Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.113.gtf.gz \
  --fasta Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.dna.toplevel.fa.gz --out ann --chroms 1-39
haplens build-genes --bfile wuji_bed --ann ann_variants.tsv --genes ann_genes.tsv --pheno pheno_fh.csv --tier A --out fhA
haplens build-genes --bfile wuji_bed --ann ann_variants.tsv --genes ann_genes.tsv --pheno pheno_fh.csv --tier B --prune-r2 0.8 --out fhB
haplens test-genes --fh fhA_fh.npz --pheno pheno_fh.csv --rel grm --traits BW0,BW8,BW16 \
  --no-sex-traits BW0 --out res_A --perm 10 --perm-genes 1000

# window layer (50 kb)
haplens make-windows --bim wuji_bed.bim --size 50000 --chroms 1-38 --out win50.blocks.det
haplens build-windows --bfile wuji_bed --blocks win50.blocks.det --pheno pheno_fh.csv --max-tags 4 --min-n 30 --out fhwin50t4
haplens test-genes --fh fhwin50t4_fh.npz --pheno pheno_fh.csv --rel grm --traits BW0,BW8,BW16 \
  --no-sex-traits BW0 --out res_win50t4 --perm 10 --perm-genes 1000
haplens test-windows --bfile wuji_bed --windows win50.blocks.det --pheno pheno_fh.csv --rel grm \
  --traits BW0,BW8,BW16 --no-sex-traits BW0 --chroms 1-38 --perm 10 --out kern
haplens combine-windows --kern kern --cls res_win50t4 --out report

# follow-up of the two loci
haplens snp-scan --bfile wuji_bed --pheno pheno_fh.csv --rel grm --traits TC16,BW16 \
  --regions 4:73000000-76700000,1:168800000-169600000 --out snp
haplens conditional --bfile wuji_bed --pheno pheno_fh.csv --rel grm --traits TC16,BW16,KL16 \
  --pairs "4:73700001-73750000=4:73734448+4:73732416" --out cond
haplens robustness --bfile wuji_bed --pheno pheno_fh.csv --rel grm --traits TC16,BW16 \
  --windows 4:73700001-73750000,1:169000001-169050000 --out loc
haplens power-windows --bfile wuji_bed --windows win50.blocks.det --pheno pheno_fh.csv --rel grm \
  --trait BW16 --n-win 200 --out power_BW16.csv
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests simulate a population of 480 birds and run the gene, window, SNP-scan, conditional and gene-check commands.

## Citation

Shen L, Zhou Y, Fan S, Zhang M, Guo J, Wei Z, Yang X. HapLens: a calibrated multi-scale haplotype association framework localises growth loci in Lueyang black-boned chickens (in preparation). See `CITATION.cff`; please also cite the exact version or commit used.

## Licence and contact

MIT licence. Linyong Shen, College of Animal Science and Technology, Northwest A&F University, Yangling, Shaanxi, China (shenlinyong@nwafu.edu.cn).
