# loqui

Local speech for Rust: Kokoro text-to-speech and Whisper speech-to-text, as
a library you link in-process or a small server that speaks the OpenAI audio
API. It is safe to embed by default.

loqui is a Rust rewrite of the parts of
[open-speech](https://github.com/jeremy-windsor/open-speech): the
OpenAI-compatible speech and transcription endpoints. It keeps the wire
contract and drops everything else.

## Status

Early development.

## Design commitments

- **No listener unless you ask for one.** As a library, loqui opens no
  sockets. `loqui serve` defaults to a unix socket in `$XDG_RUNTIME_DIR` with
  mode 0600 and a peer-uid check.
- **Exposure is explicit.** TCP binds are loopback, a named interface, or all
  interfaces with an acknowledgement flag. Any non-loopback bind also needs TLS
  or an explicit plaintext override.
- **Always a token over TCP.** If none is configured, one is generated and
  written with mode 0600. It is accepted only as `Authorization: Bearer`,
  compared in constant time, and never logged.
- **GPL-free by default.** Phonemization uses the misaki lexicons, a POS
  tagger and a small embedded neural G2P model for unknown words. eSpeak NG is
  an opt-in feature.

## License

MIT. See `LICENSE` and `NOTICE` for third-party attributions.
