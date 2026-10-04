//! A speaker-embedding network on ONNX Runtime.
//!
//! Built for WeSpeaker's models as sherpa-onnx exports them (ResNet34-LM is the
//! one loqui pins): input `feats`, f32 `[1, T, 80]`, the mean-subtracted
//! filterbank of [`crate::fbank`]; output `embs`, f32 `[1, D]`. The export's
//! own metadata says how it wants its input, and [`SpeakerEncoder::load`]
//! checks it rather than trusting a file name.

use std::path::Path;
use std::sync::Mutex;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::Error;
use crate::fbank::{Fbank, MEL_BINS, SAMPLE_RATE, subtract_mean};

/// The fewest frames an embedding is computed from: half a second. A shorter
/// clip says little about a voice, and the network's pooling has nothing to
/// pool.
pub const MIN_FRAMES: usize = 50;

pub struct SpeakerEncoder {
    // `Session::run` takes `&mut self`.
    session: Mutex<Session>,
    fbank: Fbank,
    dim: usize,
}

impl std::fmt::Debug for SpeakerEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeakerEncoder").field("dim", &self.dim).finish_non_exhaustive()
    }
}

impl SpeakerEncoder {
    /// Loads an ONNX speaker model. `threads` caps intra-op parallelism;
    /// `None` lets ONNX Runtime decide.
    ///
    /// The model must declare a 16 kHz sample rate and samples on the 16-bit
    /// scale (`normalize_samples = 0`), the features this crate computes; one
    /// that declares otherwise is refused rather than fed features it was not
    /// trained on.
    pub fn load(path: &Path, threads: Option<usize>) -> Result<Self, Error> {
        let ort_err = |e: ort::Error| Error::Model(e.to_string());
        let mut builder = Session::builder().map_err(ort_err)?;
        builder = builder.with_optimization_level(GraphOptimizationLevel::Level3).map_err(|e| Error::Model(e.to_string()))?;
        if let Some(n) = threads {
            builder = builder.with_intra_threads(n).map_err(|e| Error::Model(e.to_string()))?;
        }
        let session = builder.commit_from_file(path).map_err(ort_err)?;
        let dim = {
            let meta = session.metadata().map_err(ort_err)?;
            let custom = |key: &str| meta.custom(key);
            if custom("sample_rate").as_deref() != Some("16000") {
                return Err(Error::Model("the speaker model does not declare a 16 kHz sample rate".into()));
            }
            if custom("normalize_samples").as_deref() != Some("0") {
                return Err(Error::Model("the speaker model does not take samples on the 16-bit scale".into()));
            }
            custom("output_dim")
                .and_then(|d| d.parse::<usize>().ok())
                .ok_or_else(|| Error::Model("the speaker model does not declare its output_dim".into()))?
        };
        Ok(Self { session: Mutex::new(session), fbank: Fbank::new(), dim })
    }

    /// The embedding's length (256 for ResNet34-LM).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The unit-length embedding of 16 kHz mono samples in `[-1, 1]`.
    pub fn embed(&self, samples: &[f32]) -> Result<Vec<f32>, Error> {
        let mut features = self.fbank.compute(samples);
        if features.frames < MIN_FRAMES {
            return Err(Error::TooShort { secs: samples.len() as f32 / SAMPLE_RATE as f32, min_secs: min_secs() });
        }
        subtract_mean(&mut features);
        let ort_err = |e: ort::Error| Error::Model(e.to_string());
        let feats = Tensor::from_array(([1usize, features.frames, MEL_BINS], features.data)).map_err(ort_err)?;
        let mut session = self.session.lock().map_err(|_| Error::Model("speaker model lock poisoned".into()))?;
        let outputs = session.run(ort::inputs!["feats" => feats]).map_err(ort_err)?;
        let (_, embedding) = outputs["embs"].try_extract_tensor::<f32>().map_err(ort_err)?;
        if embedding.len() != self.dim {
            return Err(Error::Model(format!("the speaker model returned {} values, not {}", embedding.len(), self.dim)));
        }
        Ok(normalize(embedding.to_vec()))
    }
}

/// [`MIN_FRAMES`] as seconds of audio.
pub fn min_secs() -> f32 {
    (400 + (MIN_FRAMES - 1) * 160) as f32 / SAMPLE_RATE as f32
}

/// Scale to unit length; a zero vector stays zero.
pub fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

/// The cosine of the angle between two embeddings; for unit vectors, their
/// dot product. 1 for the same voice, near 0 for unrelated ones.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norms = a.iter().map(|x| x * x).sum::<f32>().sqrt() * b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norms > 0.0 { dot / norms } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_and_normalize() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 2.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 1.0], &[-1.0, -1.0]) + 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        let unit = normalize(vec![3.0, 4.0]);
        assert!((unit[0] - 0.6).abs() < 1e-6 && (unit[1] - 0.8).abs() < 1e-6);
        assert_eq!(normalize(vec![0.0, 0.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn half_a_second_is_the_floor() {
        assert!((min_secs() - 0.515).abs() < 1e-6);
    }

    #[test]
    fn a_missing_model_is_a_model_error() {
        let err = SpeakerEncoder::load(Path::new("/nonexistent/speaker.onnx"), None).unwrap_err();
        assert!(matches!(err, Error::Model(_)), "{err:?}");
    }
}
