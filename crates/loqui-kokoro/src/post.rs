//! open-speech's output post-processing (`src/audio/postprocessing.py`):
//! drop leading and trailing near-silence, then scale to a fixed peak.

/// Samples at or below this magnitude count as silence at the edges.
pub const TRIM_THRESHOLD: f32 = 0.01;
/// The peak level output is scaled to.
pub const PEAK: f32 = 0.95;

/// Keeps the span from the first to the last sample louder than
/// `threshold`. All-quiet audio is returned unchanged.
pub fn trim_silence(audio: &[f32], threshold: f32) -> Vec<f32> {
    let loud = |s: &&f32| s.abs() > threshold;
    match (audio.iter().position(|s| loud(&s)), audio.iter().rposition(|s| loud(&s))) {
        (Some(first), Some(last)) => audio[first..=last].to_vec(),
        _ => audio.to_vec(),
    }
}

/// Scales so the loudest sample reaches `peak`, clipping to `[-1, 1]`.
/// Near-silent audio is left alone.
pub fn normalize_peak(mut audio: Vec<f32>, peak: f32) -> Vec<f32> {
    let max = audio.iter().fold(0f32, |m, s| m.max(s.abs()));
    if max <= 1e-8 {
        return audio;
    }
    let gain = peak / max;
    for s in &mut audio {
        *s = (*s * gain).clamp(-1.0, 1.0);
    }
    audio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_only_the_edges() {
        assert_eq!(trim_silence(&[0.0, 0.005, 0.5, 0.0, 0.2, 0.001], 0.01), [0.5, 0.0, 0.2]);
        assert_eq!(trim_silence(&[0.0, 0.001], 0.01), [0.0, 0.001]);
        assert!(trim_silence(&[], 0.01).is_empty());
    }

    #[test]
    fn normalises_to_the_peak() {
        let out = normalize_peak(vec![0.1, -0.5, 0.25], 0.95);
        assert!((out[1] + 0.95).abs() < 1e-6);
        assert_eq!(normalize_peak(vec![0.0, 0.0], 0.95), [0.0, 0.0]);
    }
}
