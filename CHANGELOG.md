# Changelog

All notable changes to loqui are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/); before 1.0, a minor version may
break the API.

## [0.3.0] - 2026-10-04

Voice matching: the blend of the stock voices that sounds most like a
recording, found by speaker-embedding similarity.

### Added

- `loqui-speaker`, a new crate: Kaldi's log mel filterbank in pure Rust (as
  WeSpeaker computes it) and `SpeakerEncoder`, a WeSpeaker speaker model on
  ONNX Runtime.
- `EngineBuilder::speaker(Some(SpeakerConfig))` enables speaker embeddings,
  adding the pinned WeSpeaker ResNet34-LM (26 MB, CC-BY-4.0) to `missing()`,
  `fetch()`, `models()`, preload and idle unload. Off by default.
- `Engine::speaker_embedding` and `SpeakerEmbedding::similarity`.
- `Engine::match_voice(&MatchRequest, progress)` returns a `BlendMatch`: the
  blend in whole percents with the accent first, its similarity, the closest
  single voice's, the ranking of every stock voice, and a speed when a
  transcript was given. `MatchProgress` reports each step and can cancel.
- `loqui match FILE` prints the match as a `voices.toml` line, transcribing
  the recording first when Whisper is built in. `loqui fetch` also fetches
  the speaker model.

### Changed

- **Breaking:** `ModelKind` has a `Speaker` variant and is now
  `#[non_exhaustive]`. `Error` gains `Speaker` and `Cancelled`.
- `GET /v1/models` lists only the models the server serves.

## [0.2.0] - 2026-10-03

Custom voices: blends can be given names, and the blends themselves are
hardened. An embedding application can also check and prepare an engine's
weights without loading them.

### Added

- Named voices. `EngineBuilder::voice(name, spec)` names a blend of
  built-in voices. Clients can then use the name like any voice, alone or
  inside another blend. A name may replace an OpenAI name such as `nova`.
  `build()` checks every name and spec.
- `loqui serve`, `speak`, `fetch` and `doctor` read named voices from a
  TOML file: `--voices FILE` or `LOQUI_VOICES_FILE`, by default
  `voices.toml` in the config directory. `doctor` flags a file that would
  stop `serve`.
- `GET /v1/audio/voices`, open-speech's voice list: built-in voices, then
  named voices with their `blend`. It is read-only and sits behind the same
  policy as every other route.
- `loqui voices` lists the same, and `Engine::voices()` returns it as
  `VoiceInfo`.
- `Engine::fetch_voices()`. `loqui fetch` now prepares every voice pack,
  not only `af_heart`, so any voice or blend works offline. `preload` also
  fetches the packs that named voices need.
- `loqui_kokoro::Blend`: a parsed voice spec, re-exported as `loqui::Blend`.
- `Engine::missing()` lists the weights an engine needs that are not cached,
  with each file's size and where it will be. It only looks: nothing is
  loaded, hashed or downloaded, and the cache is not created, so it suits a
  health check.
- `Engine::fetch()` downloads and verifies those files without loading a
  model. `loqui fetch` uses it, so preparing a cache no longer holds the
  models in memory.
- `loqui doctor` reports how many model files are not cached. With
  `--offline`, that is a warning.
- `TranscribeRequest::threads`: the CPU threads whisper.cpp uses for one
  transcription.
- `AudioError`, `TtsError` and `SttError`: the error types `Error` wraps,
  re-exported so callers can match on the cause.
- `WhisperModel` and `whisper_model(name)`: each known Whisper model's file,
  digest and size, and whether it is English-only.
- The `whisper-cuda` feature: CUDA for Whisper only, with Kokoro left on the
  CPU.
- The `unguarded-opus` feature, for builds that patch opus-rs with
  restsend/opus-rs#31. Opus then runs on CPUs with AVX but no FMA instead of
  being refused.

### Changed

- **Breaking:**
  - `Kokoro::speak` and `speak_raw` take a `&Blend` instead of a `&str`.
  - `parse_voice_spec` is replaced by `Blend::parse` and `str::parse`.
  - `Engine::voices()` returns `Result<Vec<VoiceInfo>, Error>`.
  - `WHISPER_MODELS` lists `WhisperModel`s rather than tuples.
  - `TranscribeRequest` has a `threads` field.
  - `loqui_kokoro::Error` is `#[non_exhaustive]`.
- Blend weights follow open-speech's grammar exactly (`2`, `0.5`).
  `1e2`, `.5` and `+1` are refused.
- OpenAI names resolve inside blends as well (`alloy+af_sky`).
- Blends are mixed for each request instead of cached. Only voice packs are
  kept, so memory no longer grows with each new weighting a client sends.
- A voice pack that cannot be read is a server error (500), not a client
  error that named a cache path.

### Fixed

- Weights large enough to overflow their sum are refused. Before, they gave
  a silent voice.
- Every voice in a request is checked before any pack is fetched. Before,
  `af_heart+bad` fetched `af_heart`, and with downloads denied it answered
  500 rather than 400.
- An unknown voice is refused without its name being echoed.
- A capability the server was built without (speech or transcription) now
  answers 404 `not_enabled`, as intended, rather than 400.
- A language an English-only Whisper model cannot transcribe is a client
  error, so the server answers 400 rather than 500.
- Weights missing from the cache stay `Error::Missing` while a failed load
  is backed off. Before, the retries in the next five seconds reported
  `Unavailable`.
- A `max_audio_secs` too large to multiply by the sample rate no longer
  overflows. Debug builds panicked; release builds wrapped and refused
  short audio as too long.

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
