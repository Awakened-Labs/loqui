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
    # blends: the same, with a blend on each side
    cargo run --release -p loqui-kokoro --example speak -- model.onnx voices/ out.wav "PHONEMES" --phonemes --raw out.f32 --voice "af_bella(2)+af_heart(1)"
    kokoro_ref.py onnx model.onnx "voices/af_bella.bin(2)+voices/af_heart.bin(1)" "PHONEMES" ref.f32
    kokoro_ref.py torch "af_bella(2)+af_heart(1)" "PHONEMES" ref.f32
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

## Speaker similarity

`speaker_spike.py` asks whether a speaker-verification model can steer a
search over Kokoro blends, which is what matching a recording needs. It runs
on the host in any Python with `numpy`, `onnxruntime`, `kaldi-native-fbank`,
`scipy` and `soundfile`, speaks with Kokoro's ONNX export (packs mixed in
numpy, as `kokoro_ref.py` does), and embeds with a model exported for
sherpa-onnx: WeSpeaker's 80-dim Kaldi fbank at 16 kHz, `feats [B,T,80]` in and
`embs [B,D]` out. Phonemes come from `phonemize`, one dialect per file:

    head -7 tools/parity/corpus/harvard.txt > parity-out/spike/sentences.txt
    phonemize < sentences.txt > phonemes-us.txt; phonemize --gb < sentences.txt > phonemes-gb.txt
    speaker_spike.py synth  KOKORO.onnx VOICES_DIR config.json phonemes-us.txt phonemes-gb.txt audio
    speaker_spike.py score  SPEAKER.onnx audio [TARGET.wav ...]
    speaker_spike.py search SPEAKER.onnx KOKORO.onnx VOICES_DIR config.json phonemes-us.txt phonemes-gb.txt audio K BUDGET

`config.json` is hexgrad/Kokoro-82M's (its `vocab`). The speaker models are
from `csukuangfj/speaker-embedding-models` at revision
`0743f301363dec56491a490f6d6cbc9d67f9a3bf`.

## Results log

### 2026-10-04: speaker similarity over Kokoro voices

28 stock voices x 5 Harvard sentences; 12 voice pairs (six within a gender
and accent, two across accents, four across genders) as targets at 25, 50
and 75% on sentences 3-4, and as a 0-100% grid in steps of 10 on the same
text and on sentences 6-7. Kokoro fp32 (pinned revision), onnxruntime 1.30,
CPU, one physical core.

| | WeSpeaker ResNet34-LM | WeSpeaker CAM++-LM |
|---|---|---|
| File (sha256) | `wespeaker_en_voxceleb_resnet34_LM.onnx`, 26.5 MB (`e9848563…9c39012`) | `wespeaker_en_voxceleb_CAM++_LM.onnx`, 29.3 MB (`e197af7e…0ec10cb2`) |
| Voice identification, leave one sentence out | 140/140 | 140/140 |
| Margin, true voice over the best other (mean / min) | 0.358 / 0.062 | 0.342 / 0.124 |
| 50/50 blend's nearest voice is a parent | 9/12 | 9/12 |
| Cosine to each parent monotone in the weight | 11/12 | 11/12 |
| Grid finds the weight within 15, same text | 36/36 (mean error 3.3) | 36/36 (3.3) |
| Grid finds the weight within 15, other text | 35/36 (5.0) | 35/36 (6.7) |
| Linear mix of the two voices' embeddings vs the blend's (mean / min cosine) | 0.802 / 0.333 | 0.818 / 0.309 |
| Embedding cost | 8.6 ms per second of audio | 8.4 ms |

The three misses on "nearest voice is a parent" are all across genders: a
50/50 male and female blend sounds most like a third, androgynous voice
(`af_alloy`, `am_eric`), and the parents rank as low as 20th. Within a gender
the dominant parent always ranks 1st or 2nd and the other within the top 7.

The search (rank by cosine to each voice, start from the least-squares mix of
the top K on a 5% lattice, then move weight between pairs while it helps,
20/10/5 points at a time), on the 50% and 75% targets, with candidates
spoken on text the target never said. "Ceiling" is the true blend spoken on
that text:

| K, budget | Similarity found (mean) | Best single voice | Ceiling | Within 0.02 of the ceiling | Syntheses | Time per match |
|---|---|---|---|---|---|---|
| 4, 36 | 0.757 | 0.616 | 0.761 | 17/24 | 28.8 | 68 s |
| 6, 48 | 0.766 | 0.616 | 0.761 | 18/24 | 45.2 | 101 s |

Within a gender the search matches or beats the ceiling, usually with the
right parents and weights within 10 points. Across genders it falls short
(worst: 0.625 against 0.722) where the parents never enter the top K. The
time is dominated by Kokoro at a real-time factor of 0.35 on one core.

Decision: ResNet34-LM, K = 4, a budget of 36. CAM++ is no better; K = 6 buys
0.009 for half as much time again.

The Rust matcher (`Engine::match_voice`, defaults), run by
`cargo test -p loqui --no-default-features --test voice_match -- --ignored
--test-threads 1` against the pinned models (Kokoro fp32, CPU, one core
shared with another build; 458 s for all four, the 28 anchors included):

| Target, on a sentence the search never speaks | Found | Similarity | Best single voice | Syntheses |
|---|---|---|---|---|
| `bm_george` | `bm_george` | 0.907 | 0.907 | 16 |
| `af_bella(70)+af_sky(30)` | `af_bella(65)+af_sky(25)+af_nova(10)` | 0.878 | 0.821 (`af_bella`) | 30 |

`the_same_voice_scores_above_other_voices` and `matching_is_deterministic`
(12 syntheses, the same blend and similarity twice) pass as well. The Rust
encoder agrees with Python onnxruntime on the same model at cosine > 0.9999
(`loqui-speaker`'s ignored `the_embedding_matches_onnxruntime_in_python`).

### 2026-10-02: voice blends

The phonemes of "The birch canoe slid on the smooth planks, and the quick
brown fox jumped over the lazy dog." were spoken three ways:
- by loqui (`examples/speak.rs --phonemes`);
- by Python onnxruntime, with the ONNX packs mixed in numpy;
- by hexgrad's PyTorch KModel, with its `.pt` packs mixed as open-speech's
  `_blend_voices` mixes them.

The references ran in a throwaway container of the open-speech image, CPU
only, with no network and its model cache mounted read-only.

| Voice | Samples (all three) | loqui vs Python ONNX | loqui vs PyTorch | ONNX vs PyTorch |
|---|---|---|---|---|
| `af_bella` (baseline) | 148,200 | 0.9991 | 0.9936 | 0.9936 |
| `af_bella(2)+af_heart(1)` | 145,200 | 0.9989 | 0.9941 | 0.9941 |

A blend agrees with the references as closely as a single voice does. The
style vector drives Kokoro's duration predictor, so equal sample counts in
all three runtimes, different from the single voice's, show that each
runtime used the same blended style.

`af_heart` was chosen because it and `af_bella` were the only packs in the
container's cache. No end-to-end WER was run for blends: it would need the
open-speech server running beside the live one, and a blend changes only
the style vector, which the comparison above measures directly.

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
