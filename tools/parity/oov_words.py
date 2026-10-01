#!/usr/bin/env python3
"""Pick out-of-vocabulary words for the G2P fallback comparison.

A word qualifies when CMUdict knows it but neither misaki lexicon does, and
it is not a regular -s/-ed/-ing form of a lexicon word (misaki's morphology
rules would cover those before any fallback runs). CMUdict's pronunciation,
mapped onto misaki's alphabet, is the reference both fallbacks are scored
against.

    oov_words.py CMUDICT US_GOLD US_SILVER [N] > corpus/oov.tsv

Output: `word<TAB>reference phonemes`, N rows (default 500), chosen by a
stable hash so the list is reproducible.
"""

import hashlib
import json
import re
import sys

# ARPAbet to misaki (American). Vowels carry stress digits; the reference is
# compared with stress stripped, so only the unstressed schwa-like cases need
# their own entries.
ARPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "W", "AY": "I",
    "EH": "ɛ", "ER": "ɜɹ", "EY": "A", "IH": "ɪ", "IY": "i", "OW": "O",
    "OY": "Y", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "ʧ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h",
    "JH": "ʤ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p",
    "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}
UNSTRESSED = {"AH0": "ə", "ER0": "əɹ"}


def to_misaki(arpabet):
    out = []
    for phone in arpabet.split():
        if phone in UNSTRESSED:
            out.append(UNSTRESSED[phone])
        else:
            out.append(ARPA[re.sub(r"\d", "", phone)])
    return "".join(out)


def stable_key(word):
    return hashlib.sha256(word.encode()).hexdigest()


def main():
    cmudict, gold_path, silver_path = sys.argv[1:4]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    lexicon = set()
    for path in (gold_path, silver_path):
        with open(path) as f:
            lexicon.update(k.lower() for k in json.load(f))

    def covered(word):
        if word in lexicon:
            return True
        for suffix, stems in (("s", [word[:-1]]), ("es", [word[:-2]]),
                              ("ed", [word[:-2], word[:-1]]), ("ing", [word[:-3], word[:-3] + "e"])):
            if word.endswith(suffix) and any(s in lexicon for s in stems):
                return True
        return False

    seen = {}
    with open(cmudict, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            word, arpa = line.split(" ", 1)
            if "(" in word or not re.fullmatch(r"[a-z]{4,14}", word):
                continue
            if word in seen or covered(word):
                continue
            seen[word] = to_misaki(arpa)

    for word in sorted(seen, key=stable_key)[:n]:
        print(f"{word}\t{seen[word]}")


if __name__ == "__main__":
    main()
