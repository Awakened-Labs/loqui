# loqui-server

A safe-by-default, OpenAI-compatible speech server around a loqui `Engine`:
`/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/translations`,
`/v1/models`, `/health`, and open-speech's read-only `/v1/audio/voices`, and
nothing else.

- A Unix socket for this user by default; TCP must be asked for, and
  network-wide TCP twice.
- A bearer token on every TCP request, loopback included.
- Requests from web pages (`Origin`) refused, `Host` checked, no CORS.
- Bodies, text, audio length, queues, header arrival and request time bounded.

See [SECURITY.md](https://github.com/Awakened-Labs/loqui/blob/main/SECURITY.md)
for the exposure model and
[docs/embedding.md](https://github.com/Awakened-Labs/loqui/blob/main/docs/embedding.md)
before embedding it.

Part of [loqui](https://github.com/Awakened-Labs/loqui): local Kokoro
text-to-speech and Whisper speech-to-text for Rust. Most programs want the
[`loqui`](https://crates.io/crates/loqui) facade rather than this crate
directly. MIT; see `NOTICE` for third-party attributions.
