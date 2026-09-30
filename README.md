# loqui

Local speech for Rust: Kokoro text-to-speech and Whisper speech-to-text, as
a library you link in-process or a small server that speaks the OpenAI audio
API. It is safe to embed by default.

loqui is a Rust rewrite of the parts of
[open-speech](https://github.com/jeremy-windsor/open-speech): the
OpenAI-compatible speech and transcription endpoints. It keeps the wire
contract and drops everything else.

## Status

Working end to end: existing OpenAI-compatible speech clients run against
`loqui serve` with no code changes. Parity with the open-speech
container it replaces is measured in `tools/parity/README.md` (G2P 0.01-0.78%
phoneme error; end-to-end WER 2.42% vs the container's 2.28%). Status and
next steps: `docs/STATUS.md`.

## Quick start

    cargo install --path crates/loqui-cli      # needs cmake + a C++ compiler for Whisper
    loqui serve                                # Unix socket, this user only
    loqui serve --listen loopback:8100         # 127.0.0.1, bearer token required
    loqui token show                           # the generated token
    loqui doctor --listen all:8100             # what would this expose, and is it allowed?

    loqui speak "Hello." -o hello.wav          # in process, no server
    loqui transcribe hello.wav

As a library, with no listener at all:

```rust
let engine = loqui::Engine::builder().build()?;
let speech = engine.speak(&loqui::SpeakRequest::new("Hello from loqui."))?;
```

Models download on first use from Hugging Face, pinned to commit revisions
and checked against SHA-256 digests (`--offline` / `Downloads::Deny` to
forbid downloads; `loqui fetch` to pre-download). GPU builds:
`--features cuda` (needs the CUDA toolkit), then `--gpu cuda`.

## API

OpenAI-compatible, and nothing else:

| Route | |
|---|---|
| `POST /v1/audio/speech` | `{model, input, voice, response_format: wav\|flac\|pcm, speed}` |
| `POST /v1/audio/transcriptions` | multipart `file`, `language`, `prompt`, `temperature`, `response_format: json\|text\|verbose_json\|srt\|vtt` |
| `POST /v1/audio/translations` | as above, into English |
| `GET /v1/models` | ids with `owned_by: loqui/tts\|loqui/stt` and a `task` field |
| `GET /health` | `{"status":"ok"}` only |

Voices: the 28 American and British Kokoro voices (`af_heart` default),
OpenAI names (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`), and blends
such as `af_bella(2)+af_sky(1)`.

## Crates

| Crate | |
|---|---|
| `loqui` | the in-process `Engine`: lazy loading, idle unload, pinned downloads |
| `loqui-server` | the HTTP front end: listeners, tokens, request policy |
| `loqui-cli` | the `loqui` command |
| `loqui-g2p` | GPL-free English G2P (a Rust port of misaki) |
| `loqui-kokoro` | Kokoro-82M on ONNX Runtime |
| `loqui-whisper` | Whisper on whisper.cpp |
| `loqui-audio` | WAV/FLAC/PCM encoding; decoding and resampling for Whisper |

## License

MIT. See `LICENSE` and `NOTICE` for third-party attributions.
