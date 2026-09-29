# Parity harness

Compares loqui against the Python open-speech container, the system it
replaces. Reference output comes from the container's own misaki 0.9.4.

    docker cp tools/parity/. open-speech:/tmp/parity/
    docker exec open-speech /opt/venv/bin/python /tmp/parity/golden.py text /tmp/parity/corpus/harvard.txt > parity-out/harvard.misaki-py.txt
    cargo run --release -p loqui-g2p --example phonemize < tools/parity/corpus/harvard.txt > parity-out/harvard.loqui.txt
    python3 tools/parity/score.py parity-out/harvard.misaki-py.txt parity-out/harvard.loqui.txt --show 10

Corpora:
- `corpus/harvard.txt`: the 720 Harvard sentences (IEEE 1969, public domain).
- `corpus/assistant.txt`: assistant-style replies (numbers, money, dates,
  times, acronyms, heteronyms).
- `corpus/oov.tsv`: 500 words CMUdict knows and misaki does not, with
  CMUdict's pronunciation mapped to misaki's alphabet (`oov_words.py`).

## Results log

2026-09-29, loqui-g2p on branch g2p/foundation:

| Test | loqui | Reference | Gate |
|---|---|---|---|
| Neural fallback vs PyTorch checkpoint (500 OOV words) | 500/500 exact | — | exact |
| OOV words vs CMUdict, PER | 16.77% (neural) | 17.46% (eSpeak NG) | ≤ 1.5× eSpeak: pass |
| Harvard vs Python misaki, PER normalised | 0.52% | — | ≤ 1%: pass |
| Assistant set vs Python misaki, PER normalised | 11.52% | — | ≤ 1%: FAIL |

The assistant-set failures are text normalization in misaki-rs: currency,
decimals with units, ordinals, years, and a space before punctuation.
