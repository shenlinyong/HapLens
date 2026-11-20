"""Command-line entry point: ``haplens <command> [options]``."""

import importlib
import sys

from haplens import __version__

COMMANDS = {
    "simulate": ("simulate", "simulate a small structured population (examples, tests)"),
    "make-windows": ("make_windows", "write fixed-size windows from a .bim file"),
    "annotate": ("annotate", "classify SNPs by predicted consequence from a gene model"),
    "build-genes": ("build_genes", "gene-level functional haplotypes and matched control haplotypes"),
    "test-genes": ("test_genes", "mixed-model test of haplotype classes (genes or windows)"),
    "build-windows": ("build_windows", "haplotype classes of tag SNPs for windows"),
    "test-windows": ("window_kernel", "regional relatedness-kernel test of windows"),
    "combine-windows": ("combine_windows", "ACAT combination of the window tests and window report"),
    "snp-scan": ("snp_scan", "single-SNP scan under the null mixed model"),
    "conditional": ("conditional", "window kernel test conditional on lead SNPs"),
    "local-pve": ("local_pve", "variance explained by a window"),
    "robustness": ("robustness", "leave-one-sire-out and sire split-half stability of windows"),
    "power-windows": ("power_windows", "power of the window tests by simulated causal SNPs"),
    "power-genes": ("power_genes", "power of the gene-level test by simulation"),
    "null-compare": ("null_compare", "compare random-effect structures of the null model"),
    "check-gene": ("check_gene", "resampling, conditional and control analyses of one gene"),
    "pc-check": ("pc_check", "relate GRM principal components to the pedigree"),
    "export-tables": (
        "export_tables",
        "window minimum SNP P, GRM principal components, gene results of regions",
    ),
}


def usage():
    width = max(len(c) for c in COMMANDS)
    lines = [f"haplens {__version__}", "", "usage: haplens <command> [options]", "", "commands:"]
    lines += [f"  {name:<{width}}  {text}" for name, (_, text) in COMMANDS.items()]
    lines += ["", "Run 'haplens <command> --help' for the options of a command."]
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0
    if argv[0] not in COMMANDS:
        print(f"haplens: unknown command '{argv[0]}'\n\n{usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(f"haplens.{COMMANDS[argv[0]][0]}")
    module.main(argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
