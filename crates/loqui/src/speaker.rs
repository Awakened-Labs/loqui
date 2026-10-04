//! Speaker embeddings through the engine: a recording in, a point in voice
//! space out.
//!
//! A recording is decoded and resampled to 16 kHz, trimmed to the stretch that
//! holds speech, and its first [`MAX_EMBED_SECS`] embedded. The trimming is a
//! plain energy gate on 20 ms frames, not a voice-activity model: it exists so
//! that leading and trailing silence, which says nothing about a voice, does
//! not dilute the embedding, and so that a near-silent upload is refused
//! rather than described.

use loqui_speaker::{SAMPLE_RATE, SpeakerEncoder};

use crate::Error;

/// How much speech is embedded: the first 30 s. Longer adds cost, not
/// accuracy.
pub(crate) const MAX_EMBED_SECS: f32 = 30.0;

/// The least speech [`crate::Engine::speaker_embedding`] describes.
pub const MIN_EMBED_SECS: f32 = 1.0;

const FRAME: usize = SAMPLE_RATE as usize / 50;

/// A voice, as a point in WeSpeaker's embedding space: a unit vector, compared
/// with [`SpeakerEmbedding::similarity`].
///
/// It says how alike two voices sound. It is not evidence of who is speaking:
/// do not use it to authenticate anyone.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerEmbedding(Vec<f32>);

impl SpeakerEmbedding {
    pub(crate) fn new(vector: Vec<f32>) -> Self {
        Self(vector)
    }

    /// The unit vector.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Cosine similarity: 1 for the same voice, near 0 for unrelated ones.
    pub fn similarity(&self, other: &Self) -> f32 {
        loqui_speaker::cosine(&self.0, &other.0)
    }
}

/// The span of `samples` from its first voiced frame to its last, and how many
/// seconds of frames within it are voiced. A frame is voiced when its RMS is at
/// least 5% of the loudest frame's (and above about -80 dBFS, so digital
/// silence is never voiced).
pub(crate) fn voiced(samples: &[f32]) -> (&[f32], f32) {
    let rms: Vec<f32> = samples.chunks(FRAME).map(|f| (f.iter().map(|s| s * s).sum::<f32>() / f.len() as f32).sqrt()).collect();
    let peak = rms.iter().copied().fold(0.0f32, f32::max);
    let threshold = (peak * 0.05).max(1e-4);
    let loud: Vec<usize> = rms.iter().enumerate().filter(|(_, r)| **r >= threshold).map(|(i, _)| i).collect();
    match (loud.first(), loud.last()) {
        (Some(&first), Some(&last)) => {
            let span = &samples[first * FRAME..((last + 1) * FRAME).min(samples.len())];
            (span, loud.len() as f32 * FRAME as f32 / SAMPLE_RATE as f32)
        }
        _ => (&samples[..0], 0.0),
    }
}

/// Embed 16 kHz samples: trim to speech, require `min_secs` of it, embed the
/// first [`MAX_EMBED_SECS`].
pub(crate) fn embed(encoder: &SpeakerEncoder, samples: &[f32], min_secs: f32) -> Result<SpeakerEmbedding, Error> {
    let (span, secs) = voiced(samples);
    if secs < min_secs {
        return Err(Error::Invalid(format!(
            "the recording holds {secs:.1} s of speech; describing a voice needs at least {min_secs:.0} s"
        )));
    }
    let span = &span[..span.len().min((MAX_EMBED_SECS * SAMPLE_RATE as f32) as usize)];
    Ok(SpeakerEmbedding::new(encoder.embed(span)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(secs: f32, amplitude: f32) -> Vec<f32> {
        (0..(secs * SAMPLE_RATE as f32) as usize).map(|i| amplitude * (i as f32 * 0.1).sin()).collect()
    }

    #[test]
    fn silence_at_either_end_is_trimmed() {
        let mut samples = vec![0.0; SAMPLE_RATE as usize];
        samples.extend(tone(2.0, 0.3));
        samples.extend(vec![0.0; SAMPLE_RATE as usize / 2]);
        let (span, secs) = voiced(&samples);
        assert!((span.len() as f32 / SAMPLE_RATE as f32 - 2.0).abs() < 0.03, "{}", span.len());
        assert!((secs - 2.0).abs() < 0.03, "{secs}");
    }

    /// A quiet stretch inside the span stays in it but does not count as
    /// speech, so a long pause cannot pass for a long recording.
    #[test]
    fn a_pause_inside_is_kept_but_not_counted() {
        let mut samples = tone(1.0, 0.3);
        samples.extend(vec![0.0; SAMPLE_RATE as usize * 3]);
        samples.extend(tone(1.0, 0.3));
        let (span, secs) = voiced(&samples);
        assert!((span.len() as f32 / SAMPLE_RATE as f32 - 5.0).abs() < 0.03);
        assert!((secs - 2.0).abs() < 0.05, "{secs}");
    }

    #[test]
    fn digital_silence_holds_no_speech() {
        assert_eq!(voiced(&[0.0; 32_000]).1, 0.0);
        assert_eq!(voiced(&[]).1, 0.0);
    }

    #[test]
    fn similarity_is_the_cosine() {
        let a = SpeakerEmbedding::new(vec![1.0, 0.0]);
        let b = SpeakerEmbedding::new(vec![0.0, 1.0]);
        assert!((a.similarity(&a) - 1.0).abs() < 1e-6);
        assert!(a.similarity(&b).abs() < 1e-6);
    }
}
