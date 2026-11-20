# Input and output formats

## Input

**Genotypes.** PLINK 1 binary files (`prefix.bed`, `.bim`, `.fam`) with biallelic SNPs. The counted allele is A1 of the `.bim`. Dosages are decoded without PLINK; missing calls are mean-imputed inside the tests.

**Phenotype table** (`--pheno`, CSV). Columns `IID`, `sex`, `batch`, `sire`, `dam` and one column per trait. `IID` must equal the second column of the `.fam` file. Empty cells are missing values. A trait is analysed in the individuals that have the trait, sex and a row in the GRM. Sex and batch enter as fixed effects (batch as indicators), `dam` defines the dam random effect and `sire` defines the strata of the permutations and of the leave-one-sire-out analyses. Traits recorded before sex can be assigned (for example at hatch) are listed in `--no-sex-traits`.

**GRM** (`--rel`). Prefix of the output of `plink --make-rel square` (`prefix.rel`, `prefix.rel.id`). 0.001 is added to the diagonal.

**Window table** (`--windows`, `--blocks`). Whitespace-separated with the PLINK `.blocks.det` columns `CHR BP1 BP2 KB NSNPS SNPS`. `make-windows` writes fixed-size windows.

**Annotation.** `annotate` reads a GTF file and the genome FASTA and writes `<out>_variants.tsv` (one row per SNP and gene with the consequence class and tier) and `<out>_genes.tsv`. Tier A: stop-gained, start-lost, splice-site, stop-lost and missense. Tier B adds splice-region, 5' and 3' UTR and upstream (2 kb) variants. Synonymous and stop-retained variants form the neutral pool.

## Output

| Command | Files |
|---|---|
| `build-genes` | `<out>_fh.npz` (genes, haplotype class per individual, control class per individual, IDs), `<out>_genes.tsv`, `<out>_fh_classes.tsv` |
| `build-windows` | `<out>_fh.npz`, `<out>_blocks.tsv` |
| `test-genes` | `<out>/<trait>.csv` (`gene_id`, `df`, `F`, `P`, `pve`, effects, `P_ctl`, `pve_ctl`, `df_ctl`) and `summary_*.csv` (variance components, inflation, counts of small P values, permutation results) |
| `test-windows` | `<out>/<trait>_<chroms>.csv` (`chr`, `start`, `end`, `n_snp`, `Q`, `P`) and `summary_*.csv` |
| `combine-windows` | `<out>/<trait>_acat.csv` (`P_kern`, `P_cls`, `P_acat`, `q_acat`), `window_report.txt` |
| `snp-scan` | `<out>/<trait>_P.npz`, `<trait>_top.csv.gz`, `<trait>_region_*.csv.gz` |
| `conditional` | `<out>/conditional.csv`, `ld_r2.csv` |
| `local-pve` | CSV with the variance components of the window |
| `robustness` | `<out>/robustness.csv`, `window_groups_adjpheno.csv`, `ld_*.npz` |
| `power-windows`, `power-genes` | CSV with the P values of every simulated effect |
| `check-gene` | `<out>/summary.txt` and the tables behind it |
