//! Audio in and out for loqui.
//!
//! Out: synthesized speech is encoded as WAV (16-bit PCM), FLAC or raw
//! 16-bit PCM, the formats an OpenAI-compatible `/audio/speech` client asks
//! for. In: uploaded audio in any common container (WAV, FLAC, MP3, Ogg
//! Vorbis, MP4/AAC) is decoded, mixed to mono and resampled for Whisper.
//! Everything here is pure Rust.

mod decode;
mod encode;

pub use decode::{WHISPER_RATE, decode, decode_mono_16k, resample};
pub use encode::encode;

/// Mono audio: samples in `[-1, 1]` at `rate` Hz.
#[derive(Clone, PartialEq)]
pub struct Pcm {
    pub samples: Vec<f32>,
    pub rate: u32,
}

impl std::fmt::Debug for Pcm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pcm").field("samples", &self.samples.len()).field("rate", &self.rate).finish()
    }
}

impl Pcm {
    pub fn duration_secs(&self) -> f64 {
        if self.rate == 0 { 0.0 } else { self.samples.len() as f64 / f64::from(self.rate) }
    }
}

/// An output format, named as the OpenAI API names it in `response_format`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    Wav,
    Flac,
    /// Headerless 16-bit little-endian PCM at the synthesis rate.
    Pcm,
}

impl Format {
    /// Parses an OpenAI `response_format`. Formats this build cannot
    /// produce (`mp3`, `opus`, `aac`) are errors that name what can be.
    pub fn parse(name: &str) -> Result<Self, Error> {
        match name {
            "wav" => Ok(Self::Wav),
            "flac" => Ok(Self::Flac),
            "pcm" => Ok(Self::Pcm),
            other => Err(Error::UnsupportedFormat(other.to_owned())),
        }
    }

    pub fn content_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Flac => "audio/flac",
            Self::Pcm => "audio/pcm",
        }
    }

    pub const ALL: [Format; 3] = [Format::Wav, Format::Flac, Format::Pcm];
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unsupported output format {0:?}; this build produces wav, flac and pcm")]
    UnsupportedFormat(String),
    #[error("encoding failed: {0}")]
    Encode(String),
    #[error("could not decode audio: {0}")]
    Decode(String),
    #[error("audio is longer than the {max_secs} s limit")]
    TooLong { max_secs: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_parse_by_openai_name() {
        assert_eq!(Format::parse("wav").unwrap(), Format::Wav);
        assert_eq!(Format::parse("flac").unwrap(), Format::Flac);
        let err = Format::parse("mp3").unwrap_err().to_string();
        assert!(err.contains("mp3") && err.contains("wav"), "{err}");
    }
}
