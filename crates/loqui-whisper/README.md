# loqui-whisper

Whisper speech-to-text on whisper.cpp, decoding the way open-speech asks of
faster-whisper (beam 5, temperature 0, optional prompt, language detection).
Input is 16 kHz mono `f32`; GGML weights are not bundled.

Building needs cmake and a C++ compiler. `cuda` and `vulkan` run on a GPU
and need those toolkits.

Part of [loqui](https://github.com/Awakened-Labs/loqui): local Kokoro
text-to-speech and Whisper speech-to-text for Rust. Most programs want the
[`loqui`](https://crates.io/crates/loqui) facade rather than this crate
directly. MIT; see `NOTICE` for third-party attributions.
