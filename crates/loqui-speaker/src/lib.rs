//! Speaker embeddings: a fixed-length vector that says what a voice sounds
//! like, independent of what it says.
//!
//! Two pieces, both small: [`fbank`], Kaldi's log mel filterbank in pure Rust
//! (the features WeSpeaker's models are trained on), and [`SpeakerEncoder`],
//! the network on ONNX Runtime. Two recordings of one voice land close
//! together, measured by [`cosine`]; that is what lets an application find
//! the blend of Kokoro voices closest to a recording.
//!
//! **An embedding is a similarity, never an identity.** It says how alike two
//! voices sound, which a recording, a synthesizer or an impression can all
//! influence; it is not evidence of who is speaking and must not be used to
//! authenticate anyone.
//!
//! Weights are not bundled: the `loqui` facade downloads the pinned model.
//! WeSpeaker's VoxCeleb models are licensed CC-BY-4.0, the licence of the
//! VoxCeleb dataset; see `NOTICE`.

pub mod encoder;
pub mod fbank;

pub use encoder::{MIN_FRAMES, SpeakerEncoder, cosine, normalize};
pub use fbank::{Fbank, Features, MEL_BINS, SAMPLE_RATE};

/// Why an embedding could not be computed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The model would not load, or would not run.
    #[error("speaker model: {0}")]
    Model(String),
    /// Too little audio to say anything about a voice.
    #[error("{secs:.1} s of audio is too short to describe a voice; {min_secs:.1} s is the least")]
    TooShort { secs: f32, min_secs: f32 },
}
