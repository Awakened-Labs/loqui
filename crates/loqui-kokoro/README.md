# loqui-kokoro

Kokoro-82M text-to-speech on ONNX Runtime, fed by
[`loqui-g2p`](https://crates.io/crates/loqui-g2p). Text is phonemized, cut
into chunks the way Kokoro's own pipeline cuts it, and synthesized to 24 kHz
mono `f32`. Weights are not bundled: the `loqui` facade downloads them,
pinned to a commit and checked by SHA-256.

Building downloads a prebuilt static ONNX Runtime (MIT); the `load-dynamic`
feature links a system `libonnxruntime` instead, and `cuda` runs on an
NVIDIA GPU.

Part of [loqui](https://github.com/Awakened-Labs/loqui): local Kokoro
text-to-speech and Whisper speech-to-text for Rust. Most programs want the
[`loqui`](https://crates.io/crates/loqui) facade rather than this crate
directly. MIT; see `NOTICE` for third-party attributions.
