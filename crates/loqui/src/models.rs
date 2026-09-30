//! Which weights loqui runs, and where they come from.
//!
//! Every download is pinned to a commit revision, so a repository changing
//! upstream cannot change what loqui runs. The large files are also checked
//! against a SHA-256 digest before use. The cache directory is created
//! private (0700 on Unix): weights are not secret, but a directory others
//! can write to would let them substitute a model.

use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo, RepoType};
use sha2::{Digest, Sha256};

use crate::Error;

/// Whether loqui may fetch missing weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Downloads {
    /// Fetch missing files from Hugging Face.
    #[default]
    Allow,
    /// Use the cache only; a missing file is an error. For air-gapped hosts
    /// and for services that should never reach the network.
    Deny,
}

pub(crate) struct Pinned {
    pub repo: &'static str,
    pub revision: &'static str,
}

pub(crate) const KOKORO_REPO: Pinned =
    Pinned { repo: "onnx-community/Kokoro-82M-v1.0-ONNX", revision: "1939ad2a8e416c0acfeecc08a694d14ef25f2231" };
#[cfg(feature = "whisper")]
pub(crate) const WHISPER_REPO: Pinned =
    Pinned { repo: "ggerganov/whisper.cpp", revision: "5359861c739e955e79d9a303bcbc70fb988958b1" };

/// Kokoro model files by precision. fp32 is the reference; the smaller
/// ones trade a little quality for memory and speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KokoroVariant {
    #[default]
    Fp32,
    Fp16,
    Quantized,
}

impl KokoroVariant {
    /// The file and its SHA-256, from the repository's LFS records at the
    /// pinned revision.
    pub(crate) fn file(self) -> (&'static str, &'static str) {
        match self {
            Self::Fp32 => ("onnx/model.onnx", "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb"),
            Self::Fp16 => ("onnx/model_fp16.onnx", "ba4527a874b42b21e35f468c10d326fdff3c7fc8cac1f85e9eb6c0dfc35c334a"),
            Self::Quantized => ("onnx/model_quantized.onnx", "fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478"),
        }
    }
}

/// Whisper models loqui knows by name: the GGML file for each and its
/// SHA-256, from the repository's LFS records at the pinned revision.
pub const WHISPER_MODELS: &[(&str, &str, &str)] = &[
    ("large-v3-turbo", "ggml-large-v3-turbo.bin", "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"),
    ("large-v3", "ggml-large-v3.bin", "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2"),
    ("medium", "ggml-medium.bin", "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208"),
    ("medium.en", "ggml-medium.en.bin", "cc37e93478338ec7700281a7ac30a10128929eb8f427dda2e865faa8f6da4356"),
    ("small", "ggml-small.bin", "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"),
    ("small.en", "ggml-small.en.bin", "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d"),
    ("base", "ggml-base.bin", "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"),
    ("base.en", "ggml-base.en.bin", "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002"),
    ("tiny", "ggml-tiny.bin", "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21"),
    ("tiny.en", "ggml-tiny.en.bin", "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f"),
];

#[cfg(feature = "whisper")]
pub(crate) fn whisper_file(model: &str) -> Result<(&'static str, &'static str), Error> {
    WHISPER_MODELS
        .iter()
        .find(|(id, ..)| *id == model)
        .map(|&(_, file, sha)| (file, sha))
        .ok_or_else(|| Error::Config(format!("unknown Whisper model {model:?}")))
}

/// The default cache: `$XDG_CACHE_HOME/loqui`, else `~/.cache/loqui`.
pub fn default_cache_dir() -> Option<PathBuf> {
    std::env::var_os("XDG_CACHE_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").filter(|v| !v.is_empty()).map(|h| PathBuf::from(h).join(".cache")))
        .map(|base| base.join("loqui"))
}

pub(crate) fn private_dir(dir: &Path) -> Result<(), Error> {
    std::fs::create_dir_all(dir).map_err(|e| Error::Io(format!("creating {}: {e}", dir.display())))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Already private is left alone, so a read-only cache (a mounted
        // volume of fetched models) still works.
        let mode = std::fs::metadata(dir).map_err(|e| Error::Io(format!("{}: {e}", dir.display())))?.permissions().mode();
        if mode & 0o777 != 0o700 {
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
                .map_err(|e| Error::Io(format!("securing {}: {e}", dir.display())))?;
        }
    }
    Ok(())
}

/// Returns the local path of `file` from a pinned repository, fetching it
/// if allowed, and checks its digest when one is pinned.
pub(crate) fn fetch(cache: &Path, pinned: &Pinned, file: &str, sha256: Option<&str>, downloads: Downloads) -> Result<PathBuf, Error> {
    private_dir(cache)?;
    let repo = Repo::with_revision(pinned.repo.to_owned(), RepoType::Model, pinned.revision.to_owned());
    let hf_cache = Cache::new(cache.to_path_buf());
    let path = match hf_cache.repo(repo.clone()).get(file) {
        Some(path) => path,
        None if downloads == Downloads::Deny => {
            return Err(Error::Missing(format!("{}@{} {file} is not in the cache and downloads are off", pinned.repo, pinned.revision)));
        }
        None => {
            tracing::info!(repo = pinned.repo, file, "downloading model file");
            let api = ApiBuilder::from_cache(hf_cache)
                .with_progress(false)
                .with_retries(3)
                .build()
                .map_err(|e| Error::Download(e.to_string()))?;
            api.repo(repo).get(file).map_err(|e| Error::Download(format!("{}/{file}: {e}", pinned.repo)))?
        }
    };
    if let Some(expected) = sha256 {
        verify(&path, expected)?;
    }
    Ok(path)
}

fn verify(path: &Path, expected: &str) -> Result<(), Error> {
    let mut file = std::fs::File::open(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    let actual: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if actual != expected {
        return Err(Error::Integrity(format!("{}: sha256 {actual}, expected {expected}", path.display())));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "whisper")]
    #[test]
    fn whisper_models_resolve_by_name() {
        assert_eq!(whisper_file("large-v3-turbo").unwrap().0, "ggml-large-v3-turbo.bin");
        assert!(whisper_file("gpt-4o-transcribe").is_err());
    }

    #[test]
    fn integrity_failures_are_reported() {
        let dir = std::env::temp_dir().join(format!("loqui-verify-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("f");
        std::fs::write(&file, b"abc").unwrap();
        assert!(verify(&file, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad").is_ok());
        assert!(matches!(verify(&file, &"0".repeat(64)), Err(Error::Integrity(_))));
        std::fs::remove_file(&file).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }
}
