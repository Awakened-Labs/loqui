# loqui-audio

Audio in and out for loqui, in pure Rust. Out: WAV (16-bit PCM), FLAC, Ogg
Opus and raw PCM, with MP3 through LAME behind the off-by-default `mp3`
feature (LGPL). In: WAV, FLAC, MP3, MP4/AAC, Ogg Vorbis, and Opus in Ogg or
WebM, mixed to mono and resampled to Whisper's 16 kHz.

Part of [loqui](https://github.com/Awakened-Labs/loqui): local Kokoro
text-to-speech and Whisper speech-to-text for Rust. Most programs want the
[`loqui`](https://crates.io/crates/loqui) facade rather than this crate
directly. MIT; see `NOTICE` for third-party attributions.
