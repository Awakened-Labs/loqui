#!/usr/bin/env python3
"""Compare loqui's tokenization with spaCy's, line by line.

    compare_tokens.py SPACY.jsonl LOQUI.jsonl [--show N]

SPACY.jsonl comes from spacy_tag.py, LOQUI.jsonl from
`phonemize --tokens`. A line matches when the token texts and the
space-after flags are identical.
"""
import json
import sys


def main():
    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 10
    spacy_rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    loqui_rows = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
    assert len(spacy_rows) == len(loqui_rows), (len(spacy_rows), len(loqui_rows))
    same = tokens = 0
    shown = 0
    for n, (s, q) in enumerate(zip(spacy_rows, loqui_rows), 1):
        s = [(t, w == " ") for t, _, w in s]
        q = [(t, sp) for t, sp in q]
        tokens += len(s)
        if s == q:
            same += 1
        elif shown < show:
            shown += 1
            i = next((i for i, (a, b) in enumerate(zip(s, q)) if a != b), min(len(s), len(q)))
            print(f"line {n}: first difference at token {i}")
            print("  spacy:", s[max(0, i - 2):i + 4])
            print("  loqui:", q[max(0, i - 2):i + 4])
    print(f"{same}/{len(spacy_rows)} lines identical ({100 * same / len(spacy_rows):.3f}%), {tokens} spaCy tokens")


if __name__ == "__main__":
    main()
