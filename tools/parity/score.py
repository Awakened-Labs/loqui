#!/usr/bin/env python3
"""Phoneme error rate between line-aligned phoneme files.

    score.py REFERENCE CANDIDATE [--strict] [--show N]

PER is the Levenshtein distance over phoneme characters divided by the
reference length, summed over all lines. misaki writes each phoneme as one
character (diphthongs are A I O W Y, affricates ʤ ʧ), so characters are
phonemes. REFERENCE may be a .tsv whose second column holds the phonemes.

Unless --strict, both sides are normalised before comparison: stress marks,
length marks and spaces are dropped, and allophones that a dictionary does
not distinguish are merged (flap ɾ and glottal ʔ to t, ᵻ to ɪ, ᵊ to ə).
That scores what a listener would call a different word, not notation.
"""

import sys

MERGE = str.maketrans({"ɾ": "t", "ʔ": "t", "ᵻ": "ɪ", "ᵊ": "ə"})
DROP = str.maketrans("", "", "ˈˌː ‍")


def normalise(s, strict):
    return s if strict else s.translate(DROP).translate(MERGE)


def distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def read(path):
    with open(path, encoding="utf-8") as f:
        rows = [line.rstrip("\n") for line in f]
    return [r.split("\t")[1] if "\t" in r else r for r in rows]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strict = "--strict" in sys.argv
    show = 0
    if "--show" in sys.argv:
        show = int(sys.argv[sys.argv.index("--show") + 1])
        args = [a for a in args if a != str(show)]
    ref, cand = read(args[0]), read(args[1])
    assert len(ref) == len(cand), f"{len(ref)} reference lines vs {len(cand)} candidate lines"

    errors = total = exact = 0
    worst = []
    for i, (r, c) in enumerate(zip(ref, cand)):
        r, c = normalise(r, strict), normalise(c, strict)
        d = distance(r, c)
        errors += d
        total += max(len(r), 1)
        exact += d == 0
        if d:
            worst.append((d / max(len(r), 1), i + 1, r, c))
    print(f"PER {100 * errors / total:.2f}%  exact {exact}/{len(ref)} ({100 * exact / len(ref):.1f}%)")
    for per, line, r, c in sorted(worst, reverse=True)[:show]:
        print(f"  line {line}: {per:.2f}  ref {r}\n  {'':>{len(str(line)) + 7}}got {c}")


if __name__ == "__main__":
    main()
