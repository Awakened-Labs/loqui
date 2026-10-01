//! Audio in and out for loqui.
//!
//! Out: synthesized speech is encoded as WAV (16-bit PCM), FLAC, Ogg Opus
//! or raw 16-bit PCM, the formats an OpenAI-compatible `/audio/speech` client asks
//! for. In: uploaded audio in any common container (WAV, FLAC, MP3, Ogg
//! Vorbis or Opus, WebM/Opus, MP4/AAC) is decoded, mixed to mono and
//! resampled for Whisper. Everything here is pure Rust.

mod decode;
mod encode;
#[cfg(feature = "mp3")]
mod mp3;
mod ogg;
mod opus;

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
///
/// Non-exhaustive because `Mp3` exists only with the `mp3` feature, and any
/// crate in a build can turn that on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Format {
    Wav,
    Flac,
    /// Opus in Ogg, as OpenAI serves `opus`.
    Opus,
    /// MP3 through LAME, with the `mp3` feature.
    #[cfg(feature = "mp3")]
    Mp3,
    /// Headerless 16-bit little-endian PCM at the synthesis rate.
    Pcm,
}

impl Format {
    /// What a request that names no format gets: MP3, as OpenAI serves,
    /// where this build can encode it; otherwise WAV.
    #[cfg(feature = "mp3")]
    pub const DEFAULT: Format = Format::Mp3;
    #[cfg(not(feature = "mp3"))]
    pub const DEFAULT: Format = Format::Wav;

    /// Parses an OpenAI `response_format`. Formats this build cannot
    /// produce (`aac`, and `mp3` without the feature) are errors that name
    /// what can be.
    pub fn parse(name: &str) -> Result<Self, Error> {
        match name {
            "wav" => Ok(Self::Wav),
            "flac" => Ok(Self::Flac),
            "opus" => Ok(Self::Opus),
            #[cfg(feature = "mp3")]
            "mp3" => Ok(Self::Mp3),
            "pcm" => Ok(Self::Pcm),
            other => Err(Error::UnsupportedFormat { name: other.to_owned(), available: Self::names() }),
        }
    }

    pub fn content_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Flac => "audio/flac",
            Self::Opus => "audio/ogg",
            #[cfg(feature = "mp3")]
            Self::Mp3 => "audio/mpeg",
            Self::Pcm => "audio/pcm",
        }
    }

    pub const ALL: &[Format] = &[
        Format::Wav,
        Format::Flac,
        Format::Opus,
        #[cfg(feature = "mp3")]
        Format::Mp3,
        Format::Pcm,
    ];

    /// The OpenAI name, as `response_format` spells it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Opus => "opus",
            #[cfg(feature = "mp3")]
            Self::Mp3 => "mp3",
            Self::Pcm => "pcm",
        }
    }

    fn names() -> String {
        Self::ALL.iter().map(|f| f.name()).collect::<Vec<_>>().join(", ")
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unsupported output format {name:?}; this build produces {available}")]
    UnsupportedFormat { name: String, available: String },
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
        assert_eq!(Format::parse("opus").unwrap(), Format::Opus);
        let err = Format::parse("aac").unwrap_err().to_string();
        assert!(err.contains("aac") && err.contains("wav, flac, opus"), "{err}");
        for format in Format::ALL {
            assert_eq!(Format::parse(format.name()).unwrap(), *format);
        }
    }
}
