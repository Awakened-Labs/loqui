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

## Reproducing the tagger

The POS tagger is distilled from spaCy's `en_core_web_sm`, which is what
misaki uses:

    python3 tools/parity/gutenberg.py gutenberg.txt          # 20 public-domain novels
    python3 tools/parity/synthetic.py 30000 1 > synthetic.txt  # assistant-style text
    # tag both inside the container (spacy_tag.py), then:
    cargo run --release -p loqui-g2p --example train_tagger -- \
        gutenberg.spacy.jsonl synthetic.spacy.jsonl --prune 2.0 \
        --out crates/loqui-g2p/data/tagger-en.bin \
        --eval parity-out/harvard.spacy.jsonl --eval parity-out/assistant.spacy.jsonl

`--prune 2.0` keeps the model at 1.4 MB (0.7 MB compressed) for about 0.01
points of phoneme error; see the log below.

## Results log

### 2026-09-29: full port of misaki en.py (branch g2p/foundation)

Phoneme error rate against Python misaki 0.9.4 with its eSpeak NG fallback,
normalised (stress and length marks, spaces and flap/glottal notation
ignored):

| Corpus | Lines | PER | Exact lines |
|---|---|---|---|
| Harvard sentences | 720 | 0.01% | 719 (99.9%) |
| Assistant set (held out) | 50 | 0.72% | 42 (84%) |
| Gutenberg sample (held out) | 4,930 | 0.78% | 3,393 (68.8%) |
| Synthetic sample | 1,500 | 1.31% | 1,164 (77.6%) |

Component checks:

| Component | Result |
|---|---|
| Tokenizer vs spaCy, 44,373 paragraphs / 1.83M tokens | 100% identical |
| Number words vs num2words 0.5.14, 7,026 vectors | 100% identical |
| Neural fallback vs PyTorch checkpoint, 500 words | 500/500 exact |
| Neural fallback vs CMUdict on 500 OOV names | 16.77% PER (eSpeak NG: 17.46%) |
| Tagger vs spaCy tags, Harvard / assistant | 96.1% / 95.9% |

The remaining differences are mostly unknown words, where the neural model
and eSpeak NG disagree (names, "café", "naïve"), and eSpeak's abbreviation
table ("St." as "Saint"). Several synthetic-set differences are upstream
bugs that loqui deliberately does not reproduce: Python misaki drops
negative numbers the tagger calls UH ("-13 degrees" is read as "degrees")
and garbles some euro amounts through eSpeak.

### 2026-09-29: first cut (misaki-rs with upstream lexicons)

Harvard 0.52%, assistant set 11.52%: misaki-rs's text normalisation read
currency, units, ordinals and years wrongly, which led to the full port.
