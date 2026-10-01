#!/usr/bin/env python3
"""Export spaCy's English tokenizer rules as data for loqui-g2p.

misaki tokenizes with spaCy's `en_core_web_sm`, and its lexicon logic is
tuned to that segmentation ("don't" is "do" + "n't", "$12.50" is "$" +
"12.50"). loqui-g2p reimplements the tokenizer algorithm in Rust and reads
the rules this script writes. spaCy and its English rules are MIT-licensed.

    spacy_export.py OUT.json

Run inside the open-speech container, which has spaCy 3.8 installed.
"""

import json
import sys

import spacy
from spacy.attrs import ORTH


def main():
    nlp = spacy.blank("en")
    tokenizer = nlp.tokenizer
    special_cases = {
        text: [piece.get(ORTH, piece.get("ORTH")) for piece in pieces]
        for text, pieces in tokenizer.rules.items()
    }
    out = {
        "spacy_version": spacy.__version__,
        "prefix": tokenizer.prefix_search.__self__.pattern,
        "suffix": tokenizer.suffix_search.__self__.pattern,
        "infix": tokenizer.infix_finditer.__self__.pattern,
        "url": tokenizer.url_match.__self__.pattern if tokenizer.url_match else None,
        "special_cases": dict(sorted(special_cases.items())),
    }
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
