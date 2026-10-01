#!/usr/bin/env python3
"""End-to-end intelligibility: speak text, transcribe it, score word errors.

    e2e.py speak      TEXT.txt OUT_DIR [--url URL] [--voice af_heart]
    e2e.py transcribe WAV_DIR  OUT.txt [--url URL]
    e2e.py wer        TEXT.txt HYP.txt [--show N]

`speak` asks an OpenAI-compatible server (default: the open-speech container
at http://127.0.0.1:8100/v1) for one WAV per line. `transcribe` sends each
NNNN.wav in a directory to the same kind of server's Whisper. `wer` scores
transcripts against the source text after normalising both (lowercase,
punctuation dropped, digits spelled out by the same simple rules), so a
number spoken correctly scores the same whether Whisper writes it in
digits or words.
"""

import json
import os
import re
import sys
import urllib.request
import uuid

URL = "http://127.0.0.1:8100/v1"


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def post_json(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def post_file(url, path, fields):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    with open(path, "rb") as f:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                     f"Content-Type: audio/wav\r\n\r\n".encode() + f.read() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(url, b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
       "sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def spell(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + spell(n % 100))
    for size, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= size:
            return spell(n // size) + " " + name + ("" if n % size == 0 else " " + spell(n % size))
    return str(n)


def normalise(text):
    text = text.lower().replace("%", " percent").replace("&", " and ").replace("$", " dollars ")
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    text = re.sub(r"\d+", lambda m: " " + spell(int(m.group())) + " ", text)
    text = re.sub(r"[^a-z' ]+", " ", text).replace("'", "")
    return text.split()


def wer(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def main():
    mode = sys.argv[1]
    url = arg("--url", URL)
    if mode == "speak":
        text, out = sys.argv[2], sys.argv[3]
        os.makedirs(out, exist_ok=True)
        for i, line in enumerate(open(text, encoding="utf-8").read().splitlines(), 1):
            audio = post_json(f"{url}/audio/speech", {"model": "kokoro", "voice": arg("--voice", "af_heart"),
                                                      "input": line, "response_format": "wav"})
            open(os.path.join(out, f"{i:04d}.wav"), "wb").write(audio)
    elif mode == "transcribe":
        wav_dir, out = sys.argv[2], sys.argv[3]
        with open(out, "w", encoding="utf-8") as f:
            for name in sorted(n for n in os.listdir(wav_dir) if n.endswith(".wav")):
                r = post_file(f"{url}/audio/transcriptions", os.path.join(wav_dir, name), {"response_format": "json"})
                f.write(r["text"].strip().replace("\n", " ") + "\n")
    elif mode == "wer":
        refs = open(sys.argv[2], encoding="utf-8").read().splitlines()
        hyps = open(sys.argv[3], encoding="utf-8").read().splitlines()
        assert len(refs) == len(hyps), (len(refs), len(hyps))
        errors = words = 0
        worst = []
        for n, (r, h) in enumerate(zip(refs, hyps), 1):
            rw, hw = normalise(r), normalise(h)
            e = wer(rw, hw)
            errors, words = errors + e, words + len(rw)
            if e:
                worst.append((e / max(len(rw), 1), n, r, h))
        print(f"WER {100 * errors / words:.2f}% ({errors}/{words} words, {len(worst)} lines with errors)")
        for rate, n, r, h in sorted(worst, reverse=True)[: int(arg("--show", 0))]:
            print(f"  {n}: {rate:.2f}\n    ref: {r}\n    hyp: {h}")


if __name__ == "__main__":
    main()
