"""Convert borrowed CUDA-Q terms into owned Python values immediately."""
import math


def flatten_ops(ops, n):
    coeffs, words = [], []
    for op in ops:
        for term in op:
            # CUDA-Q iterators can reuse a borrowed term. Never collect term objects.
            coeffs.append(float(term.evaluate_coefficient().real))
            words.append(str(term.get_pauli_word(n)))
    return coeffs, words


def validate_pool(pool, n):
    """Check every constrained pool entry, before spending time on training."""
    for index, op in enumerate(pool):
        coeffs, words = flatten_ops([op], n)
        if not all(math.isfinite(c) for c in coeffs):
            raise ValueError(f"Pool entry {index}: non-finite coefficient")
        if any(len(w) != n or set(w) - set("IXYZ") for w in words):
            raise ValueError(f"Pool entry {index}: invalid Pauli word")
        if len(words) == 1 and set(words[0]) <= set("IZ"):
            continue
        support = [tuple(i for i, p in enumerate(w) if p != "I") for w in words]
        families = ["".join(p for p in w if p != "I") for w in words]
        if (len(words) != 2 or sorted(families) != ["XX", "YY"]
                or support[0] != support[1] or len(support[0]) != 2
                or not math.isclose(coeffs[0], coeffs[1], rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"Pool entry {index}: expected a matched XX+YY pair, got {list(zip(coeffs, words))}")


def validate_sequence(terms):
    """Check serialized constrained circuits, including historical result files."""
    i = 0
    while i < len(terms):
        word = terms[i]["pauli_word"]
        if set(word) <= set("IZ"):
            i += 1
            continue
        if i + 1 >= len(terms):
            raise ValueError("Unpaired mixer term")
        other = terms[i + 1]["pauli_word"]
        support = lambda w: tuple(j for j, p in enumerate(w) if p != "I")
        family = lambda w: "".join(p for p in w if p != "I")
        if (sorted((family(word), family(other))) != ["XX", "YY"]
                or support(word) != support(other)
                or abs(terms[i]["coeff"] - terms[i + 1]["coeff"]) > 1e-12):
            raise ValueError(f"Broken XX+YY mixer pair: {word}, {other}")
        i += 2
