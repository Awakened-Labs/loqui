//! Kaldi's log mel filterbank, as WeSpeaker computes it for its speaker models.
//!
//! WeSpeaker trains on `torchaudio.compliance.kaldi.fbank` with 80 mel bins,
//! 25 ms frames every 10 ms, no dither, a Hamming window and no energy term,
//! then subtracts each bin's mean over the utterance; sherpa-onnx feeds its
//! ONNX exports the same features through kaldi-native-fbank. This is that
//! computation, in Rust, held to kaldi-native-fbank by
//! `tests/fbank_parity.rs`. Each step, in order:
//!
//! - samples on the 16-bit scale (the export's `normalize_samples = 0`);
//! - frames of 400 samples every 160, the last partial frame dropped
//!   (`snip_edges`), so `1 + (n - 400) / 160` frames;
//! - per frame: the mean subtracted (`remove_dc_offset`), pre-emphasis 0.97
//!   with the first sample scaled by `1 - 0.97`, a Hamming window
//!   `0.54 - 0.46 cos(2 pi i / 399)`, zero padding to 512;
//! - the power spectrum, bins 0 to 255 (Kaldi's mel banks skip the Nyquist
//!   bin);
//! - 80 triangular filters evenly spaced on Kaldi's mel scale
//!   `1127 ln(1 + f / 700)` between 20 Hz and 8 kHz, each weight taken at the
//!   bin's own mel value;
//! - the natural log, floored at `f32::EPSILON`.
//!
//! [`subtract_mean`] is the last step, kept apart so the parity test can hold
//! the filterbank to its reference before it.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

/// The sample rate the features are defined for.
pub const SAMPLE_RATE: u32 = 16_000;
/// Mel bins per frame.
pub const MEL_BINS: usize = 80;

const FRAME_LENGTH: usize = 400;
const FRAME_SHIFT: usize = 160;
const FFT_SIZE: usize = 512;
const PREEMPHASIS: f32 = 0.97;
const LOW_FREQ: f32 = 20.0;
const SAMPLE_SCALE: f32 = 32_768.0;

/// Log mel energies: `frames` rows of [`MEL_BINS`], row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Features {
    pub frames: usize,
    pub data: Vec<f32>,
}

impl Features {
    /// Row `i`.
    pub fn frame(&self, i: usize) -> &[f32] {
        &self.data[i * MEL_BINS..(i + 1) * MEL_BINS]
    }
}

/// How many frames `samples` samples make: whole frames only.
pub fn frame_count(samples: usize) -> usize {
    if samples < FRAME_LENGTH { 0 } else { 1 + (samples - FRAME_LENGTH) / FRAME_SHIFT }
}

/// The filterbank. Built once and reused: the window, the mel weights and the
/// FFT plan do not change between calls.
pub struct Fbank {
    window: Vec<f32>,
    /// Per mel bin: the first FFT bin it weighs, and the weights from there.
    banks: Vec<(usize, Vec<f32>)>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl std::fmt::Debug for Fbank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fbank").finish_non_exhaustive()
    }
}

impl Default for Fbank {
    fn default() -> Self {
        Self::new()
    }
}

fn mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

impl Fbank {
    pub fn new() -> Self {
        let window = (0..FRAME_LENGTH)
            .map(|i| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (FRAME_LENGTH - 1) as f64).cos() as f32)
            .collect();
        // Kaldi's `MelBanks`: centres evenly spaced in mel, weights read off
        // at each FFT bin's mel value; the Nyquist bin is not among them.
        let bin_width = SAMPLE_RATE as f32 / FFT_SIZE as f32;
        let (mel_low, mel_high) = (mel(LOW_FREQ), mel(SAMPLE_RATE as f32 / 2.0));
        let delta = (mel_high - mel_low) / (MEL_BINS + 1) as f32;
        let banks = (0..MEL_BINS)
            .map(|b| {
                let left = mel_low + b as f32 * delta;
                let centre = left + delta;
                let right = centre + delta;
                let mut first = None;
                let mut weights = Vec::new();
                for i in 0..FFT_SIZE / 2 {
                    let m = mel(bin_width * i as f32);
                    if m > left && m < right {
                        first.get_or_insert(i);
                        weights.push(if m <= centre { (m - left) / (centre - left) } else { (right - m) / (right - centre) });
                    }
                }
                (first.unwrap_or(0), weights)
            })
            .collect();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE);
        Self { window, banks, fft }
    }

    /// Log mel energies of 16 kHz mono samples in `[-1, 1]`.
    pub fn compute(&self, samples: &[f32]) -> Features {
        let frames = frame_count(samples.len());
        let mut data = Vec::with_capacity(frames * MEL_BINS);
        let mut buffer = vec![0.0f32; FFT_SIZE];
        let mut spectrum = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut power = vec![0.0f32; FFT_SIZE / 2];
        for f in 0..frames {
            let frame = &samples[f * FRAME_SHIFT..f * FRAME_SHIFT + FRAME_LENGTH];
            let mean = frame.iter().map(|s| s * SAMPLE_SCALE).sum::<f32>() / FRAME_LENGTH as f32;
            for (out, s) in buffer.iter_mut().zip(frame) {
                *out = s * SAMPLE_SCALE - mean;
            }
            for i in (1..FRAME_LENGTH).rev() {
                buffer[i] -= PREEMPHASIS * buffer[i - 1];
            }
            buffer[0] -= PREEMPHASIS * buffer[0];
            for (s, w) in buffer.iter_mut().zip(&self.window) {
                *s *= w;
            }
            buffer[FRAME_LENGTH..].fill(0.0);
            self.fft.process_with_scratch(&mut buffer, &mut spectrum, &mut scratch).expect("buffers sized by the plan");
            for (p, c) in power.iter_mut().zip(&spectrum) {
                *p = c.norm_sqr();
            }
            data.extend(self.banks.iter().map(|(first, weights)| {
                let energy: f32 = weights.iter().zip(&power[*first..]).map(|(w, p)| w * p).sum();
                energy.max(f32::EPSILON).ln()
            }));
        }
        Features { frames, data }
    }
}

/// Subtract each mel bin's mean over the frames: the utterance-level
/// normalization WeSpeaker applies before its models.
pub fn subtract_mean(features: &mut Features) {
    if features.frames == 0 {
        return;
    }
    let mut mean = [0.0f64; MEL_BINS];
    for row in features.data.chunks_exact(MEL_BINS) {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += f64::from(*v);
        }
    }
    let frames = features.frames as f64;
    for row in features.data.chunks_exact_mut(MEL_BINS) {
        for (v, m) in row.iter_mut().zip(&mean) {
            *v -= (m / frames) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_whole_and_every_ten_milliseconds() {
        assert_eq!(frame_count(0), 0);
        assert_eq!(frame_count(399), 0);
        assert_eq!(frame_count(400), 1);
        assert_eq!(frame_count(559), 1);
        assert_eq!(frame_count(560), 2);
        assert_eq!(frame_count(16_000), 98);
        assert_eq!(Fbank::new().compute(&vec![0.1; 16_000]).frames, 98);
    }

    /// Every bin weighs something, the triangles rise to at most 1, and they
    /// march up the spectrum from 20 Hz without reaching the Nyquist bin.
    #[test]
    fn the_mel_banks_are_kaldis_triangles() {
        let fbank = Fbank::new();
        let mut previous_first = 0;
        for (b, (first, weights)) in fbank.banks.iter().enumerate() {
            assert!(!weights.is_empty(), "bank {b} is empty");
            assert!(weights.iter().all(|w| *w > 0.0 && *w <= 1.0), "bank {b}");
            assert!(*first >= previous_first, "bank {b} starts below bank {}", b.saturating_sub(1));
            assert!(first + weights.len() <= FFT_SIZE / 2, "bank {b} reaches the Nyquist bin");
            previous_first = *first;
        }
        // 20 Hz sits between FFT bins 0 (0 Hz) and 1 (31.25 Hz).
        assert_eq!(fbank.banks[0].0, 1);
    }

    /// Silence floors at log(EPSILON) rather than producing -inf.
    #[test]
    fn silence_is_floored_not_infinite() {
        let features = Fbank::new().compute(&[0.0; 800]);
        assert_eq!(features.frames, 3);
        assert!(features.data.iter().all(|v| (*v - f32::EPSILON.ln()).abs() < 1e-6));
    }

    #[test]
    fn subtracting_the_mean_zeroes_each_bin() {
        let samples: Vec<f32> = (0..8_000).map(|i| (i as f32 * 0.07).sin() * 0.3).collect();
        let mut features = Fbank::new().compute(&samples);
        subtract_mean(&mut features);
        for b in 0..MEL_BINS {
            let sum: f32 = (0..features.frames).map(|f| features.frame(f)[b]).sum();
            assert!(sum.abs() / (features.frames as f32) < 1e-4, "bin {b}: {sum}");
        }
    }
}
