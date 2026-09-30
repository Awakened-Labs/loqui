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

## Synthesis and end to end

    # same phonemes through three runtimes
    cargo run --release -p loqui-kokoro --example speak -- model.onnx voices/ out.wav "PHONEMES" --phonemes --raw out.f32
    docker exec open-speech /opt/venv/bin/python /tmp/parity/kokoro_ref.py onnx ... / torch ...
    # intelligibility: speak with both systems, transcribe with one Whisper, score
    python3 tools/parity/e2e.py speak corpus.txt container-wavs/
    cargo run --release -p loqui-kokoro --example speak -- model.onnx voices/ loqui-wavs/ --lines corpus.txt
    python3 tools/parity/e2e.py transcribe container-wavs/ container.txt   # likewise loqui-wavs
    python3 tools/parity/e2e.py wer corpus.txt container.txt

## Speech to text

The container's audio for the corpus, transcribed by loqui-whisper on the
GPU inside the CUDA dev image (`docker/cuda.Dockerfile`, target `toolchain`):

    docker run --rm --gpus all -v "$PWD":/src:ro -v loqui-cuda13-target:/target \
        -v loqui-cuda-registry:/usr/local/cargo/registry loqui:cuda-dev \
        cargo build --release --locked -p loqui-whisper --features cuda --example transcribe
    docker run --rm --gpus all -v ~/.cache/loqui:/models:ro -v "$PWD/parity-out":/data \
        -v loqui-cuda13-target:/target loqui:cuda-dev /target/release/examples/transcribe \
        /models/.../ggml-large-v3-turbo.bin --gpu 0 --language en --dir /data/container-wavs --out /data/stt-loqui.txt
    python3 tools/parity/e2e.py wer corpus.txt stt-loqui.txt

## Results log

### 2026-09-30: CUDA (docker/cuda.Dockerfile, RTX 2070 Super)

Whisper: the 250 container-synthesised clips (552 s) from the end-to-end run
below, large-v3-turbo, beam 5, temperature 0, English:

| Transcriber | WER vs the text | Real-time factor |
|---|---|---|
| faster-whisper (open-speech) | 2.28% | not measured |
| loqui-whisper, whisper.cpp on CUDA 13.0 | 2.19% | 0.158 |
| loqui-whisper, CPU (laptop) | not run (too slow) | ~8 |

The two transcribers differ on 0.37% of words (8 of 2,144). What remains
against the text is formatting both share: "6:00 p.m." for "6 PM",
"third" for "3rd", compounds ("bluefish", "drugstore"). A CUDA 12.8 build
gave the same transcript, word for word (real-time factor 0.162).

Kokoro, in the CUDA 13.0.2 image:

| Check | Result |
|---|---|
| GPU vs CPU output, same text (6.8 s) | same length, correlation 0.998 |
| `loqui serve --gpu cuda`, warm request | 4.46 s of audio in 0.41 s (real-time factor 0.09; CPU 0.30) |
| `loqui serve --gpu cuda`, warm transcription of a 2.4 s clip | 0.64 s |
| GPU memory, Kokoro fp32 alone / with Whisper large-v3-turbo | 1.1 GB / 2.7 GB |

### 2026-09-29: Kokoro on ONNX Runtime (loqui-kokoro)

Same phoneme string, voice af_heart, speed 1 (4.95 s of audio):

| Comparison | Length | Correlation |
|---|---|---|
| loqui (Rust `ort`) vs Python onnxruntime, same ONNX file | identical (118,800) | 0.997 |
| loqui vs hexgrad PyTorch KModel (what open-speech serves) | identical | 0.986 |

Not bit-exact because the vocoder injects random noise; the durations the
model predicts match exactly. Output level differs from PyTorch before
peak normalisation, which open-speech and loqui both apply.

End to end, 250 lines (200 Harvard + 50 assistant), each system's audio
transcribed by the container's faster-whisper large-v3-turbo:

| System | WER |
|---|---|
| open-speech (PyTorch Kokoro, misaki + eSpeak NG) | 2.28% |
| loqui (ONNX Kokoro, loqui-g2p) | 2.42% |

+0.14 points; the M0 gate was +0.5. Most residual "errors" are shared
scoring artifacts (compounds, "3rd" vs "third"). loqui synthesised the 552 s
of audio in 168 s on the laptop CPU (real-time factor 0.30).

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
