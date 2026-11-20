#!/bin/bash
# Window and gene layers on a simulated population of 480 birds (about 10 s).
set -euo pipefail
out=${1:-example}
d=$out/data

haplens simulate --out "$d"

# gene layer
haplens build-genes --bfile "$d/ex" --ann "$d/ann_variants.tsv" --genes "$d/ann_genes.tsv" \
  --pheno "$d/pheno.csv" --tier A --out "$out/genesA"
haplens test-genes --fh "$out/genesA_fh.npz" --pheno "$d/pheno.csv" --rel "$d/grm" \
  --out "$out/res_genes" --perm 10 --perm-genes 20

# window layer
haplens build-windows --bfile "$d/ex" --blocks "$d/windows.det" --pheno "$d/pheno.csv" \
  --max-tags 4 --min-n 30 --out "$out/win"
haplens test-genes --fh "$out/win_fh.npz" --pheno "$d/pheno.csv" --rel "$d/grm" --out "$out/res_win"
haplens test-windows --bfile "$d/ex" --windows "$d/windows.det" --pheno "$d/pheno.csv" \
  --rel "$d/grm" --chroms 1 --perm 5 --out "$out/kern"
haplens combine-windows --kern "$out/kern" --cls "$out/res_win" --out "$out/report"

# single-SNP scan and a conditional window test
haplens snp-scan --bfile "$d/ex" --pheno "$d/pheno.csv" --rel "$d/grm" --traits T1 \
  --regions 1:100000-400000 --out "$out/snp"
haplens conditional --bfile "$d/ex" --pheno "$d/pheno.csv" --rel "$d/grm" --traits T1 \
  --pairs "1:1399000-1438000=1:1420000" --out "$out/cond"
