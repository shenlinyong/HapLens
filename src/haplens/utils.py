"""Small helpers shared by the command modules."""


def chrom_list(spec):
    """Expand a chromosome specification such as '1-38' or '1-5,Z' into a list of names."""
    out = []
    for token in spec.split(","):
        token = token.strip()
        if "-" in token and token.replace("-", "").isdigit():
            lo, hi = map(int, token.split("-"))
            out += [str(i) for i in range(lo, hi + 1)]
        elif token:
            out.append(token)
    return out
