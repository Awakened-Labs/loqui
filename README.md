# loqui

Local speech for Rust: Kokoro text-to-speech and Whisper speech-to-text, as
a library you link in-process or a small server that speaks the OpenAI audio
API. It is safe to embed by default.

loqui began as the speech backend for ennius, our agent runtime: a Rust
rewrite of the parts of
[open-speech](https://github.com/jeremy-windsor/open-speech) by
[jeremy-windsor](https://github.com/jeremy-windsor) that ennius used, the
OpenAI-compatible speech and transcription endpoints. It keeps the wire
contract and drops everything else.

## Status

0.1: working end to end. Existing OpenAI-compatible speech clients run
against `loqui serve` with no code changes. Parity with the open-speech
container it replaces is measured in `tools/parity/README.md`:
- G2P: 0.01-0.78% phoneme error;
- end-to-end WER: 2.42%, against the container's 2.28%;
- transcription WER: 2.19%, against faster-whisper's 2.28%.

What is verified and what is still open is in `docs/STATUS.md`; changes are
in `CHANGELOG.md`.

## Quick start

    cargo install loqui-cli                    # needs cmake + a C++ compiler for Whisper
    loqui serve                                # Unix socket, this user only
    loqui serve --listen loopback:8100         # 127.0.0.1, bearer token required
    loqui token show                           # the generated token
    loqui doctor --listen all:8100             # what would this expose, and is it allowed?

    loqui speak "Hello." -o hello.wav          # in process, no server
    loqui transcribe hello.wav

As a library, with no listener at all (`cargo add loqui`, plus
`--features whisper` for speech-to-text):

```rust
let engine = loqui::Engine::builder().build()?;
let speech = engine.speak(&loqui::SpeakRequest::new("Hello from loqui."))?;
```

Models download on first use from Hugging Face, pinned to commit revisions
and checked against SHA-256 digests (`--offline` / `Downloads::Deny` to
forbid downloads; `loqui fetch` to pre-download). GPU builds:
`--features cuda` (needs the CUDA toolkit), then `--gpu cuda`.

MP3 output is off by default: `--features mp3` builds it in with LAME, which
is LGPL (a C compiler and make at build time). With it, `response_format`
may be `mp3` and defaults to it, as OpenAI's does; without it, the default
is `wav`. Everything else in loqui is MIT and permissively licensed.

## API

OpenAI-compatible, plus open-speech's voice list, and nothing else:

| Route | |
|---|---|
| `POST /v1/audio/speech` | `{model, input, voice, response_format: wav\|flac\|opus\|pcm (\|mp3), speed}` |
| `POST /v1/audio/transcriptions` | multipart `file`, `language`, `prompt`, `temperature`, `response_format: json\|text\|verbose_json\|srt\|vtt` |
| `POST /v1/audio/translations` | as above, into English |
| `GET /v1/models` | ids with `owned_by: loqui/tts\|loqui/stt` and a `task` field |
| `GET /v1/audio/voices` | `{"voices": [{id, name, language, gender}]}` as open-speech lists them; named voices add `blend` |
| `GET /health` | `{"status":"ok"}` only |

### Voices

The 28 American and British Kokoro voices (`af_heart` is the default), and
OpenAI's names for six of them (`alloy`, `echo`, `fable`, `onyx`, `nova`,
`shimmer`). Any of them can be mixed, as in open-speech:
`af_bella(2)+af_sky(1)` is two parts Bella to one part Sky. The first voice
sets the accent, even at weight 0, so `bf_emma(0)+af_sky` is Sky with a
British accent.

Give the blends you use names in `voices.toml`, in loqui's config directory
(`~/.config/loqui/`), or in a file passed with `--voices`. Any client can
then ask for them by name, alone or inside another blend:

```toml
will = "am_puck(1)+am_liam(1)+am_onyx(0.5)"
nova = "nova(3)+af_sky(1)"   # replaces OpenAI's nova, for clients that offer only OpenAI's six
```

A name is lowercase letters, digits and `_`; Kokoro-shaped names (`af_*`)
are kept for Kokoro's own voices. `loqui voices` lists every voice, named
ones included, and `loqui doctor` checks the file. Names are read when the
server starts, never over HTTP.

## Crates

| Crate | |
|---|---|
| `loqui` | the in-process `Engine`: lazy loading, idle unload, pinned downloads |
| `loqui-server` | the HTTP front end: listeners, tokens, request policy |
| `loqui-cli` | the `loqui` command |
| `loqui-g2p` | GPL-free English G2P (a Rust port of misaki) |
| `loqui-kokoro` | Kokoro-82M on ONNX Runtime |
| `loqui-whisper` | Whisper on whisper.cpp |
| `loqui-audio` | WAV, FLAC, Ogg Opus and PCM out (MP3 behind a feature); decoding and resampling for Whisper |

## Security

loqui is safe to embed by default, and every widening of its exposure is a
named, deliberate choice:
- [SECURITY.md](SECURITY.md) describes the exposure model and how to report
  a vulnerability privately.
- [docs/embedding.md](docs/embedding.md) covers running loqui inside your
  own program: limits, tokens, and the licensing each feature brings.

## License

MIT. See `LICENSE` and `NOTICE` for third-party attributions.
