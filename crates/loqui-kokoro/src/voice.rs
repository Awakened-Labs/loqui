//! Kokoro voice packs.
//!
//! A voice is 510 style vectors of 256 floats, one per possible phoneme
//! count: Kokoro conditions each utterance on the row matching its length.
//! The `.bin` files in `onnx-community/Kokoro-82M-v1.0-ONNX` are exactly
//! that array as little-endian f32, the same data as hexgrad's `.pt` packs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use crate::{Blend, Error};

pub const STYLE_DIM: usize = 256;
/// Rows in a pack, and so the longest phoneme sequence Kokoro accepts.
pub const MAX_PHONEMES: usize = 510;
const PACK_LEN: usize = MAX_PHONEMES * STYLE_DIM;

#[derive(Clone)]
pub struct Voice {
    pack: Vec<f32>,
}

impl std::fmt::Debug for Voice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Voice").finish_non_exhaustive()
    }
}

impl Voice {
    /// A malformed pack is the installation's fault, not the caller's, so it
    /// is a [`Error::Model`] error.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() != PACK_LEN * 4 {
            return Err(Error::Model(format!("a voice pack is {} bytes, got {}", PACK_LEN * 4, bytes.len())));
        }
        let pack = bytes.as_chunks::<4>().0.iter().copied().map(f32::from_le_bytes).collect();
        Ok(Self { pack })
    }

    pub fn from_file(path: &Path) -> Result<Self, Error> {
        let bytes = std::fs::read(path).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
        Self::from_bytes(&bytes)
    }

    /// A weighted average of voices, as open-speech blends
    /// `af_bella(2)+af_sky(1)`. Weights are normalised to sum to one; if
    /// they sum to zero, the voices are weighted equally.
    pub fn blend(parts: &[(&Voice, f32)]) -> Result<Self, Error> {
        let Some(&(first, _)) = parts.first() else {
            return Err(Error::Voice("a blend needs at least one voice".into()));
        };
        let total: f32 = parts.iter().map(|(_, w)| w).sum();
        if parts.iter().any(|(_, w)| !(w.is_finite() && *w >= 0.0)) || !total.is_finite() {
            return Err(Error::Voice("voice weights must be non-negative and not too large".into()));
        }
        let mut pack = vec![0.0; first.pack.len()];
        for (voice, weight) in parts {
            let w = if total == 0.0 { 1.0 / parts.len() as f32 } else { weight / total };
            for (p, v) in pack.iter_mut().zip(&voice.pack) {
                *p += w * v;
            }
        }
        Ok(Self { pack })
    }

    /// The style vector for an utterance of `phonemes` phonemes: row
    /// `phonemes - 1`, as `KPipeline.infer` indexes `pack[len(ps)-1]`.
    pub fn style(&self, phonemes: usize) -> &[f32] {
        let row = phonemes.clamp(1, MAX_PHONEMES) - 1;
        &self.pack[row * STYLE_DIM..(row + 1) * STYLE_DIM]
    }
}

/// The packs in a voices directory, each read once.
///
/// Only packs are kept. A blend is mixed afresh on every call, which costs
/// a few hundred thousand multiply-adds beside a synthesis of billions, so
/// what is held is bounded by the files in the directory however many
/// weightings callers ask for.
pub(crate) struct VoicePacks {
    dir: PathBuf,
    loaded: Mutex<HashMap<String, Arc<Voice>>>,
}

impl VoicePacks {
    pub(crate) fn new(dir: PathBuf) -> Self {
        Self { dir, loaded: Mutex::new(HashMap::new()) }
    }

    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    /// The voice `blend` describes. Parts weighing nothing are never read.
    pub(crate) fn get(&self, blend: &Blend) -> Result<Arc<Voice>, Error> {
        let packs = {
            // A panic elsewhere cannot leave a half-written entry: inserts
            // are whole `Arc`s, so the map is usable after a poisoning.
            let mut loaded = self.loaded.lock().unwrap_or_else(PoisonError::into_inner);
            blend.audible().map(|(id, weight)| Ok((self.load(&mut loaded, id)?, weight))).collect::<Result<Vec<_>, Error>>()?
        };
        match packs.as_slice() {
            [(voice, _)] => Ok(Arc::clone(voice)),
            _ => {
                let parts: Vec<(&Voice, f32)> = packs.iter().map(|(voice, weight)| (voice.as_ref(), *weight)).collect();
                Ok(Arc::new(Voice::blend(&parts)?))
            }
        }
    }

    fn load(&self, loaded: &mut HashMap<String, Arc<Voice>>, id: &str) -> Result<Arc<Voice>, Error> {
        if let Some(voice) = loaded.get(id) {
            return Ok(Arc::clone(voice));
        }
        let voice = Arc::new(Voice::from_file(&self.dir.join(format!("{id}.bin")))?);
        loaded.insert(id.to_owned(), Arc::clone(&voice));
        Ok(voice)
    }

    #[cfg(test)]
    fn cached(&self) -> usize {
        self.loaded.lock().unwrap_or_else(PoisonError::into_inner).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant(value: f32) -> Voice {
        Voice { pack: vec![value; PACK_LEN] }
    }

    #[test]
    fn rejects_packs_of_the_wrong_size() {
        assert!(Voice::from_bytes(&[0; 12]).is_err());
        assert!(Voice::from_bytes(&vec![0; PACK_LEN * 4]).is_ok());
    }

    #[test]
    fn style_rows_follow_phoneme_count() {
        let mut v = constant(0.0);
        v.pack[STYLE_DIM * 9] = 1.0;
        assert_eq!(v.style(10)[0], 1.0);
        assert_eq!(v.style(0).len(), STYLE_DIM);
        assert_eq!(v.style(10_000).len(), STYLE_DIM);
    }

    #[test]
    fn blends_are_normalised_weighted_means() {
        let (a, b) = (constant(1.0), constant(4.0));
        let mixed = Voice::blend(&[(&a, 2.0), (&b, 1.0)]).unwrap();
        assert!((mixed.style(1)[0] - 2.0).abs() < 1e-6);
        let equal = Voice::blend(&[(&a, 0.0), (&b, 0.0)]).unwrap();
        assert!((equal.style(1)[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn blends_reject_weights_that_are_not_proportions() {
        let (a, b) = (constant(1.0), constant(4.0));
        for weights in [(-1.0, 1.0), (f32::NAN, 1.0), (f32::INFINITY, 1.0), (3e38, 3e38)] {
            assert!(Voice::blend(&[(&a, weights.0), (&b, weights.1)]).is_err(), "{weights:?}");
        }
    }

    /// A voices directory of constant packs, removed on drop.
    struct Packs(PathBuf);

    impl Packs {
        fn new(name: &str, voices: &[(&str, f32)]) -> Self {
            let dir = std::env::temp_dir().join(format!("loqui-packs-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            for (id, value) in voices {
                let bytes: Vec<u8> = std::iter::repeat_n(value.to_le_bytes(), PACK_LEN).flatten().collect();
                std::fs::write(dir.join(format!("{id}.bin")), bytes).unwrap();
            }
            Self(dir)
        }
    }

    impl Drop for Packs {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn only_packs_are_cached_however_many_weightings_are_asked_for() {
        let dir = Packs::new("bounded", &[("af_bella", 1.0), ("af_sky", 4.0)]);
        let packs = VoicePacks::new(dir.0.clone());
        for weight in 1..=50 {
            let blend: Blend = format!("af_bella({weight})+af_sky(1)").parse().unwrap();
            let expected = (weight as f32 + 4.0) / (weight as f32 + 1.0);
            assert!((packs.get(&blend).unwrap().style(1)[0] - expected).abs() < 1e-5);
        }
        assert_eq!(packs.cached(), 2);
    }

    #[test]
    fn a_single_voice_is_shared_not_copied() {
        let dir = Packs::new("shared", &[("af_sky", 4.0)]);
        let packs = VoicePacks::new(dir.0.clone());
        let blend: Blend = "af_sky".parse().unwrap();
        assert!(Arc::ptr_eq(&packs.get(&blend).unwrap(), &packs.get(&blend).unwrap()));
    }

    #[test]
    fn weightless_parts_are_never_read() {
        // bf_emma has no pack here; it only sets the accent.
        let dir = Packs::new("weightless", &[("af_sky", 4.0)]);
        let packs = VoicePacks::new(dir.0.clone());
        let voice = packs.get(&"bf_emma(0)+af_sky".parse().unwrap()).unwrap();
        assert_eq!(voice.style(1)[0], 4.0);
        assert_eq!(packs.cached(), 1);
    }

    #[test]
    fn a_missing_pack_is_a_model_error_not_a_voice_error() {
        let dir = Packs::new("missing", &[]);
        let packs = VoicePacks::new(dir.0.clone());
        assert!(matches!(packs.get(&"af_sky".parse().unwrap()), Err(Error::Model(_))));
    }
}
