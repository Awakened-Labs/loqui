#!/usr/bin/env python3
"""Fetch the public-domain novels the tagger is trained on, as paragraphs.

    gutenberg.py OUT.txt

Downloads 20 Project Gutenberg books (all public domain in the US), strips
the Gutenberg header and licence footer, and writes one paragraph of 20 to
600 characters per line. The book list is fixed so training is repeatable.
"""

import re
import sys
import urllib.request

BOOKS = [1342, 1661, 2701, 74, 84, 98, 11, 345, 36, 35, 1400, 76, 5200, 174, 2600, 4300, 1260, 768, 158, 120]


def main():
    paragraphs = []
    for book in BOOKS:
        url = f"https://www.gutenberg.org/cache/epub/{book}/pg{book}.txt"
        with urllib.request.urlopen(url) as r:
            text = r.read().decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
        start = re.search(r"\*\*\* ?START OF[^\n]*\n", text)
        end = re.search(r"\*\*\* ?END OF", text)
        if start and end:
            text = text[start.end():end.start()]
        for p in re.split(r"\n\s*\n", text):
            p = re.sub(r"\s+", " ", p).strip().replace("_", "")
            if 20 <= len(p) <= 600:
                paragraphs.append(p)
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        f.write("\n".join(paragraphs) + "\n")


if __name__ == "__main__":
    main()
