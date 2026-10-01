# Changelog

All notable changes to loqui are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/); before 1.0, a minor version may
break the API.

## [0.1.0] - 2026-10-01

The first release: local Kokoro text-to-speech and Whisper speech-to-text
for Rust, as a library or as a server speaking the OpenAI audio API.

### Added

- `loqui`: the in-process `Engine`. It loads models lazily and can unload
  them after an idle time. Downloads are pinned to Hugging Face commit
  revisions and checked against SHA-256 digests; `Downloads::Deny` forbids
  them. Input text and audio length are bounded.
- `loqui-g2p`: a GPL-free Rust port of misaki's English G2P. It has the
  spaCy tokenizer, a distilled POS tagger, and an embedded neural model for
  out-of-vocabulary words in place of eSpeak NG.
- `loqui-kokoro`: Kokoro-82M on ONNX Runtime. It has the 28 American and
  British voices, OpenAI voice aliases, and weighted voice blends.
- `loqui-whisper`: Whisper on whisper.cpp, decoding as faster-whisper does
  for open-speech.
- `loqui-audio`:
  - out: WAV, FLAC, Ogg Opus and raw PCM, with MP3 through LAME behind
    the off-by-default `mp3` feature (LGPL);
  - in: WAV, FLAC, MP3, MP4/AAC, Ogg Vorbis, and Opus in Ogg or WebM;
  - resampling to 16 kHz for Whisper.
- `loqui-server`: `/v1/audio/speech`, `/v1/audio/transcriptions`,
  `/v1/audio/translations`, `/v1/models` and `/health`, safe by default:
  - a Unix socket for the owner only;
  - bearer tokens on every TCP request;
  - explicit acknowledgement for network and all-interface listeners, and
    TLS or an explicit plaintext opt-in off loopback;
  - `Origin` refused, `Host` checked, no CORS;
  - bounded bodies, queues, connections, header arrival and request time.
- `loqui-cli`: the `loqui` command, with `serve`, `speak`, `transcribe`,
  `fetch`, `token` and `doctor`.
- GPU inference with the `cuda` feature (CUDA 13), and a container image in
  `docker/cuda.Dockerfile`.
- `SECURITY.md` (the exposure model and private reporting), and
  `docs/embedding.md`.
