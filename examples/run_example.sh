#!/bin/bash
# End-to-end HapLens v1.0.0 example on a simulated structured population (480 birds, ~15 s).
set -euo pipefail
out=${1:-example}
d=$out/data

haplens simulate --out "$d"

# One-command pipeline (HapLens-O windows + gene-level Bi-LBC specificity test)
haplens run --bfile "$d/ex" --pheno "$d/pheno.csv" --rel "$d/grm" \
  --traits T0,T1 --chroms 1 --window-size 50000 --perm 5 \
  --ann "$d/ann_variants.tsv" --genes "$d/ann_genes.tsv" --tier A --out "$out/run_out"

# Single-SNP scan and conditional dissection
haplens snp-scan --bfile "$d/ex" --pheno "$d/pheno.csv" --rel "$d/grm" --traits T1 \
  --regions 1:100000-400000 --out "$out/snp"
haplens conditional --bfile "$d/ex" --pheno "$d/pheno.csv" --rel "$d/grm" --traits T1 \
  --pairs "1:1399000-1438000=1:1420000" --out "$out/cond"
