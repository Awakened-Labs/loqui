//! Kokoro voice packs.
//!
//! A voice is 510 style vectors of 256 floats, one per possible phoneme
//! count: Kokoro conditions each utterance on the row matching its length.
//! The `.bin` files in `onnx-community/Kokoro-82M-v1.0-ONNX` are exactly
//! that array as little-endian f32, the same data as hexgrad's `.pt` packs.

use std::path::Path;

use crate::Error;

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
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() != PACK_LEN * 4 {
            return Err(Error::Voice(format!("a voice pack is {} bytes, got {}", PACK_LEN * 4, bytes.len())));
        }
        let pack = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        Ok(Self { pack })
    }

    pub fn from_file(path: &Path) -> Result<Self, Error> {
        let bytes = std::fs::read(path).map_err(|e| Error::Voice(format!("{}: {e}", path.display())))?;
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
}
