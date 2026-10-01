//! The Kokoro-82M network on ONNX Runtime.
//!
//! One call maps a phoneme string (at most 510 phonemes), a style vector and
//! a speed to 24 kHz mono audio. The graph is `onnx-community`'s export:
//! inputs `input_ids` (i64 `[1, T]`, padded with 0 at both ends), `style`
//! (f32 `[1, 256]`) and `speed` (f32 `[1]`); output `waveform`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{LazyLock, Mutex};

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::Error;
use crate::voice::{MAX_PHONEMES, STYLE_DIM, Voice};

/// Output sample rate.
pub const SAMPLE_RATE: u32 = 24_000;

/// Phoneme to token id, from hexgrad/Kokoro-82M's `config.json` (the table
/// `KModel` uses; 114 symbols, id 0 reserved for padding).
static VOCAB: LazyLock<HashMap<char, i64>> = LazyLock::new(|| {
    let table: HashMap<String, i64> = serde_json::from_str(include_str!("../data/vocab.json")).expect("embedded vocab is valid JSON");
    table.into_iter().filter_map(|(k, v)| Some((k.chars().next()?, v))).collect()
});

/// Where inference runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    /// A CUDA GPU by ordinal. Needs the `cuda` feature; without it, or
    /// without a usable CUDA runtime, loading fails rather than silently
    /// running on the CPU.
    Cuda(i32),
}

pub struct KokoroModel {
    // `Session::run` takes `&mut self`; one inference at a time per model
    // is also what a single GPU wants.
    session: Mutex<Session>,
    device: Device,
}

impl std::fmt::Debug for KokoroModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KokoroModel").field("device", &self.device).finish_non_exhaustive()
    }
}

impl KokoroModel {
    /// Loads an ONNX file. `threads` caps intra-op parallelism on the CPU;
    /// `None` lets ONNX Runtime decide.
    pub fn load(path: &Path, device: Device, threads: Option<usize>) -> Result<Self, Error> {
        let ort_err = |e: ort::Error| Error::Model(e.to_string());
        let mut builder = Session::builder().map_err(ort_err)?;
        builder = builder.with_optimization_level(GraphOptimizationLevel::Level3).map_err(|e| Error::Model(e.to_string()))?;
        if let Some(n) = threads {
            builder = builder.with_intra_threads(n).map_err(|e| Error::Model(e.to_string()))?;
        }
        match device {
            Device::Cpu => {}
            #[cfg(feature = "cuda")]
            Device::Cuda(ordinal) => {
                use ort::ep::{CUDA, ExecutionProvider};
                let cuda = CUDA::default().with_device_id(ordinal);
                if !cuda.is_available().unwrap_or(false) {
                    return Err(Error::Model("CUDA was requested but ONNX Runtime has no usable CUDA provider".into()));
                }
                builder = builder.with_execution_providers([cuda.build().error_on_failure()]).map_err(|e| Error::Model(e.to_string()))?;
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                return Err(Error::Model("CUDA was requested but loqui-kokoro was built without the `cuda` feature".into()));
            }
        }
        let session = builder.commit_from_file(path).map_err(ort_err)?;
        Ok(Self { session: Mutex::new(session), device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Synthesizes one chunk. Phonemes past the 510th are dropped, with the
    /// same truncation `KPipeline` applies; symbols outside Kokoro's
    /// vocabulary are skipped, as `KModel` skips them.
    pub fn synthesize(&self, phonemes: &str, voice: &Voice, speed: f32) -> Result<Vec<f32>, Error> {
        let phonemes: String = phonemes.chars().take(MAX_PHONEMES).collect();
        let count = phonemes.chars().count();
        if count == 0 {
            return Ok(Vec::new());
        }
        let ids: Vec<i64> =
            std::iter::once(0).chain(phonemes.chars().filter_map(|c| VOCAB.get(&c).copied())).chain(std::iter::once(0)).collect();
        let ort_err = |e: ort::Error| Error::Model(e.to_string());
        let input_ids = Tensor::from_array(([1usize, ids.len()], ids)).map_err(ort_err)?;
        let style = Tensor::from_array(([1usize, STYLE_DIM], voice.style(count).to_vec())).map_err(ort_err)?;
        let speed = Tensor::from_array(([1usize], vec![speed])).map_err(ort_err)?;

        let mut session = self.session.lock().map_err(|_| Error::Model("model lock poisoned".into()))?;
        let outputs = session.run(ort::inputs!["input_ids" => input_ids, "style" => style, "speed" => speed]).map_err(ort_err)?;
        let (_, samples) = outputs["waveform"].try_extract_tensor::<f32>().map_err(ort_err)?;
        Ok(samples.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_covers_the_misaki_alphabets() {
        assert_eq!(VOCAB.len(), 114);
        for c in "AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʤʧˈˌθᵊᵻʔTQaɒː ;:,.!?—…\"()“”".chars() {
            assert!(VOCAB.contains_key(&c), "{c:?} missing");
        }
        assert!(!VOCAB.contains_key(&'$'));
    }
}
