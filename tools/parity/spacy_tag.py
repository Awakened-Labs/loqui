#!/usr/bin/env python3
"""Tokenize and tag text with spaCy `en_core_web_sm`, the pipeline misaki uses.

    spacy_tag.py IN.txt OUT.jsonl

One input line per output line: a JSON list of [text, tag, whitespace]
triples. The output serves twice: it is the reference the Rust tokenizer is
checked against, and the training data the loqui-g2p tagger is distilled
from. en_core_web_sm is MIT-licensed; its predicted tags carry no other
licence.
"""

import json
import sys
import warnings

warnings.filterwarnings("ignore")

import spacy


def main():
    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
    with open(sys.argv[1], encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    with open(sys.argv[2], "w", encoding="utf-8") as out:
        for doc in nlp.pipe(lines, batch_size=256):
            row = [[t.text, t.tag_, t.whitespace_] for t in doc]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
