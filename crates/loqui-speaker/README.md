# loqui-speaker

Speaker embeddings: a fixed-length vector that says what a voice sounds like,
whatever it is saying. Kaldi's log mel filterbank in pure Rust, and
WeSpeaker's ResNet34-LM on ONNX Runtime, so two recordings of one voice can be
compared by cosine similarity. loqui uses it to find the blend of Kokoro
voices closest to a recording.

An embedding measures how alike two voices sound. It is not evidence of who is
speaking, and must not be used to authenticate anyone.

Weights are not bundled: the `loqui` facade downloads the pinned model.
WeSpeaker's VoxCeleb models are CC-BY-4.0.

Part of [loqui](https://github.com/Awakened-Labs/loqui): local Kokoro
text-to-speech and Whisper speech-to-text for Rust. Most programs want the
[`loqui`](https://crates.io/crates/loqui) facade rather than this crate
directly. MIT; see `NOTICE` for third-party attributions.
