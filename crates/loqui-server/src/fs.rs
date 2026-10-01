//! Private files and the default places loqui keeps them.

use std::path::{Path, PathBuf};

use crate::Error;

/// The effective user id of this process.
#[cfg(unix)]
pub fn current_uid() -> u32 {
    rustix::process::geteuid().as_raw()
}

/// Creates `dir` if needed and restricts it to its owner (0700 on Unix).
/// An existing directory owned by someone else is refused.
pub fn private_dir(dir: &Path) -> Result<(), Error> {
    std::fs::create_dir_all(dir).map_err(|e| Error::Io(format!("creating {}: {e}", dir.display())))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let meta = std::fs::symlink_metadata(dir).map_err(|e| Error::Io(format!("{}: {e}", dir.display())))?;
        if !meta.is_dir() {
            return Err(Error::Io(format!("{} is not a directory", dir.display())));
        }
        if meta.uid() != current_uid() {
            return Err(Error::Io(format!("{} is owned by uid {}, not by this user", dir.display(), meta.uid())));
        }
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
            .map_err(|e| Error::Io(format!("securing {}: {e}", dir.display())))?;
    }
    Ok(())
}

fn env_dir(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).filter(|v| !v.is_empty()).map(PathBuf::from).filter(|p| p.is_absolute())
}

/// `$XDG_CONFIG_HOME/loqui`, else `~/.config/loqui`.
pub fn config_dir() -> Option<PathBuf> {
    env_dir("XDG_CONFIG_HOME").or_else(|| env_dir("HOME").map(|h| h.join(".config"))).map(|d| d.join("loqui"))
}

/// Where the generated token lives.
pub fn default_token_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("token"))
}

/// `$XDG_RUNTIME_DIR/loqui/loqui.sock`. There is deliberately no fallback
/// to `/tmp`: a shared directory is where socket hijacking happens.
pub fn default_socket_path() -> Option<PathBuf> {
    env_dir("XDG_RUNTIME_DIR").map(|d| d.join("loqui").join("loqui.sock"))
}
