#!/usr/bin/env python3
"""Reference phonemes from Python misaki, run inside the open-speech container.

    golden.py text   FILE [--gb]   misaki + eSpeak NG fallback (Kokoro's real pipeline)
    golden.py lexeme FILE [--gb]   misaki with no fallback (lexicon and rules only)
    golden.py bart   DIR  FILE     the PyTorch G2P checkpoint in DIR, greedy, per word

One output line per input line (for .tsv input, the first column is used).
Run it with the container's interpreter, for example:

    docker cp tools/parity open-speech:/tmp/parity
    docker exec open-speech /opt/venv/bin/python /tmp/parity/golden.py text /tmp/parity/corpus/harvard.txt
"""

import sys
import warnings

warnings.filterwarnings("ignore")


def lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n").split("\t")[0]


def misaki_g2p(british, with_fallback):
    from misaki import en, espeak

    fallback = espeak.EspeakFallback(british=british) if with_fallback else None
    return en.G2P(trf=False, british=british, fallback=fallback)


def bart(model_dir, path):
    import torch
    from transformers import BartForConditionalGeneration

    model = BartForConditionalGeneration.from_pretrained(model_dir).eval()
    graphemes = model.config.grapheme_chars
    phonemes = model.config.phoneme_chars
    ids = {c: i for i, c in enumerate(graphemes) if i >= 4}
    for word in lines(path):
        encoded = [1] + [ids[c] for c in word if c in ids] + [2]
        with torch.no_grad():
            out = model.generate(
                input_ids=torch.tensor([encoded]),
                max_length=64,
                num_beams=1,
                do_sample=False,
            )
        print("".join(phonemes[t] for t in out[0].tolist() if t > 3 and t < len(phonemes)))


def main():
    mode = sys.argv[1]
    if mode == "bart":
        bart(sys.argv[2], sys.argv[3])
        return
    british = "--gb" in sys.argv
    g2p = misaki_g2p(british, with_fallback=(mode == "text"))
    for text in lines(sys.argv[2]):
        phonemes, _ = g2p(text)
        print(phonemes)


if __name__ == "__main__":
    main()
