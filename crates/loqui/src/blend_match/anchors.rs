//! The stock voices' embeddings: the search's map of voice space.
//!
//! Each voice speaks [`PROBE`] in its own accent and is embedded. That takes
//! one synthesis per voice, so the result is kept in the cache, under a key
//! that changes whenever anything that went into it does: the Kokoro weights,
//! the speaker model, the probe, the roster, and [`VERSION`] for the code in
//! between. A stale or damaged file is never read as anchors: its key no
//! longer matches, or its length is wrong, and it is computed again.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::{Error, models};

/// What each voice says to be embedded: two Harvard sentences (IEEE, 1969,
/// public domain), about six seconds of speech.
pub(crate) const PROBE: &str = "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background.";

/// Bumped when the computation changes without its inputs changing: the
/// filterbank, the trimming, how the probe is spoken.
const VERSION: &str = "anchors-v1";

/// The cache key for anchors computed from these inputs.
pub(crate) fn key(kokoro_sha256: &str, speaker_sha256: &str, voices: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for part in [VERSION, models::KOKORO_REPO.revision, kokoro_sha256, speaker_sha256, PROBE].into_iter().chain(voices.iter().copied()) {
        hasher.update(part.as_bytes());
        hasher.update([0]);
    }
    hasher.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// Where anchors with `key` are kept.
pub(crate) fn path(cache: &Path, key: &str) -> PathBuf {
    cache.join("anchors").join(format!("{key}.f32"))
}

/// `count` anchors of `dim` values, if the file holds exactly that.
pub(crate) fn read(path: &Path, count: usize, dim: usize) -> Option<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() != count * dim * 4 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(dim * 4)
            .map(|row| row.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
            .collect(),
    )
}

/// Keep `anchors` at `path`: little-endian f32, row by row, written to a
/// temporary name and renamed into place, so a reader never sees half a file.
pub(crate) fn write(path: &Path, anchors: &[Vec<f32>]) -> Result<(), Error> {
    let dir = path.parent().ok_or_else(|| Error::Io(format!("{} has no parent", path.display())))?;
    models::private_dir(dir)?;
    let bytes: Vec<u8> = anchors.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
    let temporary = path.with_extension(format!("f32.{}", std::process::id()));
    std::fs::write(&temporary, bytes).map_err(|e| Error::Io(format!("{}: {e}", temporary.display())))?;
    std::fs::rename(&temporary, path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_changes_with_every_input() {
        let base = key("k", "s", &["af_a", "am_b"]);
        assert_eq!(base.len(), 16);
        assert_eq!(base, key("k", "s", &["af_a", "am_b"]), "deterministic");
        for other in
            [key("K", "s", &["af_a", "am_b"]), key("k", "S", &["af_a", "am_b"]), key("k", "s", &["af_a"]), key("k", "s", &["am_b", "af_a"])]
        {
            assert_ne!(base, other);
        }
    }

    #[test]
    fn anchors_round_trip_and_a_wrong_length_is_not_read() {
        let dir = std::env::temp_dir().join(format!("loqui-anchors-{}", std::process::id()));
        let path = path(&dir, "abc");
        let anchors = vec![vec![0.5, -0.25, 1.0], vec![0.0, 2.0, -1.5]];
        write(&path, &anchors).unwrap();
        assert_eq!(read(&path, 2, 3), Some(anchors));
        assert_eq!(read(&path, 3, 3), None, "a roster of another size");
        assert_eq!(read(&path, 2, 4), None, "a model of another dimension");
        std::fs::remove_file(&path).unwrap();
        assert_eq!(read(&path, 2, 3), None, "absent");
        std::fs::remove_dir(path.parent().unwrap()).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }
}
