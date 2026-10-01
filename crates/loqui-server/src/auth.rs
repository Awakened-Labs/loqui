//! Bearer tokens.
//!
//! Only SHA-256 digests of accepted tokens are held in memory, and a
//! presented token is compared against each in constant time. Plaintext is
//! read, hashed and zeroed. A token is accepted only in an
//! `Authorization: Bearer` header, never in a query string, where it would
//! land in logs and browser history.
//!
//! Where tokens come from, in order: an explicit token file, the
//! `LOQUI_TOKEN` environment variable (discouraged: other processes of the
//! same user can read `/proc/<pid>/environ`), and otherwise a token loqui
//! generates on first run and keeps in a file only the owner can read.
//! Token files must be regular files owned by the current user with no
//! group or other permissions; anything else is refused rather than
//! trusted.

use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

use crate::Error;

/// Generated tokens start with this, so a leaked one is recognisable in a
/// secret scanner.
pub const TOKEN_PREFIX: &str = "loqui_";

/// Where the accepted token came from, for `loqui doctor` and the startup
/// log. Never the token itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenSource {
    File(PathBuf),
    Environment,
    Generated(PathBuf),
}

impl std::fmt::Display for TokenSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File(p) => write!(f, "token file {}", p.display()),
            Self::Environment => f.write_str("LOQUI_TOKEN environment variable"),
            Self::Generated(p) => write!(f, "generated token in {}", p.display()),
        }
    }
}

/// The set of tokens a server accepts.
pub struct Tokens {
    digests: Vec<[u8; 32]>,
    source: TokenSource,
}

impl std::fmt::Debug for Tokens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokens").field("count", &self.digests.len()).field("source", &self.source).finish()
    }
}

fn digest(token: &str) -> [u8; 32] {
    Sha256::digest(token.as_bytes()).into()
}

impl Tokens {
    /// Accepts exactly `token`. For tests and embedding.
    pub fn from_plaintext(token: &str, source: TokenSource) -> Result<Self, Error> {
        validate(token)?;
        Ok(Self { digests: vec![digest(token)], source })
    }

    /// Resolves tokens in precedence order: `file`, then `LOQUI_TOKEN`,
    /// then the generated token at `generated_path` (created if missing).
    pub fn resolve(file: Option<&Path>, env: Option<Zeroizing<String>>, generated_path: &Path) -> Result<Self, Error> {
        if let Some(path) = file {
            return Self::load_file(path, TokenSource::File(path.to_owned()));
        }
        if let Some(token) = env {
            let token = token.trim();
            validate(token)?;
            return Ok(Self { digests: vec![digest(token)], source: TokenSource::Environment });
        }
        if !generated_path.exists() {
            generate_file(generated_path)?;
        }
        Self::load_file(generated_path, TokenSource::Generated(generated_path.to_owned()))
    }

    /// Reads one token per non-empty, non-`#` line, so a file can hold a
    /// token per client and tokens can be rotated without downtime.
    fn load_file(path: &Path, source: TokenSource) -> Result<Self, Error> {
        check_private_file(path)?;
        let text = Zeroizing::new(std::fs::read_to_string(path).map_err(|e| Error::Token(format!("reading {}: {e}", path.display())))?);
        let mut digests = Vec::new();
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            validate(line)?;
            digests.push(digest(line));
        }
        if digests.is_empty() {
            return Err(Error::Token(format!("{} holds no token", path.display())));
        }
        Ok(Self { digests, source })
    }

    pub fn source(&self) -> &TokenSource {
        &self.source
    }

    /// Whether `presented` is an accepted token. Every stored digest is
    /// compared, in constant time, whichever matches.
    pub fn accepts(&self, presented: &str) -> bool {
        let presented = digest(presented);
        self.digests.iter().fold(subtle::Choice::from(0), |any, d| any | d.ct_eq(&presented)).into()
    }

    /// Checks an `Authorization` header value.
    pub fn accepts_header(&self, header: Option<&str>) -> bool {
        header.and_then(|h| h.strip_prefix("Bearer ").or_else(|| h.strip_prefix("bearer "))).is_some_and(|token| self.accepts(token.trim()))
    }
}

/// Tokens must be long enough not to be guessed and printable enough to go
/// in a header.
fn validate(token: &str) -> Result<(), Error> {
    if token.len() < 16 {
        return Err(Error::Token("a token must be at least 16 characters".into()));
    }
    if !token.bytes().all(|b| b.is_ascii_graphic()) {
        return Err(Error::Token("a token must be printable ASCII without spaces".into()));
    }
    Ok(())
}

/// 256 bits from the operating system's RNG, base64url without padding.
pub fn generate() -> Result<Zeroizing<String>, Error> {
    let mut bytes = Zeroizing::new([0u8; 32]);
    getrandom::fill(bytes.as_mut()).map_err(|e| Error::Token(format!("OS random source failed: {e}")))?;
    Ok(Zeroizing::new(format!("{TOKEN_PREFIX}{}", base64url(bytes.as_ref()))))
}

fn base64url(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
        for i in 0..=chunk.len() {
            out.push(ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char);
        }
    }
    out
}

/// Writes a fresh token to `path` with mode 0600, atomically: it is written
/// to a temporary file in the same private directory, then renamed, so a
/// crash never leaves a half-written or world-readable token.
pub fn generate_file(path: &Path) -> Result<(), Error> {
    let dir = path.parent().ok_or_else(|| Error::Token(format!("{} has no parent directory", path.display())))?;
    crate::fs::private_dir(dir)?;
    let token = generate()?;
    let tmp = dir.join(format!(".token.{}.tmp", std::process::id()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let result = options.open(&tmp).and_then(|mut f| {
        writeln!(f, "{}", token.as_str())?;
        f.sync_all()
    });
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(Error::Token(format!("writing {}: {e}", tmp.display())));
    }
    std::fs::rename(&tmp, path).map_err(|e| Error::Token(format!("installing {}: {e}", path.display())))?;
    Ok(())
}

/// Replaces the token file with a new token. Existing clients stop working.
pub fn rotate(path: &Path) -> Result<(), Error> {
    if path.exists() {
        check_private_file(path)?;
    }
    let dir = path.parent().ok_or_else(|| Error::Token(format!("{} has no parent directory", path.display())))?;
    let staged = dir.join(".token.rotate");
    let _ = std::fs::remove_file(&staged);
    generate_file(&staged)?;
    std::fs::rename(&staged, path).map_err(|e| Error::Token(format!("installing {}: {e}", path.display())))
}

/// Refuses a token file others could read or replace.
pub fn check_private_file(path: &Path) -> Result<(), Error> {
    let meta = std::fs::symlink_metadata(path).map_err(|e| Error::Token(format!("{}: {e}", path.display())))?;
    if !meta.file_type().is_file() {
        return Err(Error::Token(format!("{} is not a regular file (symlinks are refused)", path.display())));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if meta.mode() & 0o077 != 0 {
            return Err(Error::Token(format!(
                "{} is readable or writable by others (mode {:o}); run: chmod 600 {}",
                path.display(),
                meta.mode() & 0o777,
                path.display()
            )));
        }
        if meta.uid() != crate::fs::current_uid() {
            return Err(Error::Token(format!("{} is owned by uid {}, not by this user", path.display(), meta.uid())));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("loqui-auth-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn generated_tokens_are_long_prefixed_and_distinct() {
        let (a, b) = (generate().unwrap(), generate().unwrap());
        assert!(a.starts_with(TOKEN_PREFIX));
        assert_eq!(a.len(), TOKEN_PREFIX.len() + 43);
        assert_ne!(*a, *b);
    }

    #[test]
    fn base64url_matches_rfc4648() {
        assert_eq!(base64url(b"f"), "Zg");
        assert_eq!(base64url(b"fo"), "Zm8");
        assert_eq!(base64url(b"foo"), "Zm9v");
        assert_eq!(base64url(&[0xfb, 0xff]), "-_8");
    }

    #[test]
    fn accepts_only_bearer_headers_with_a_known_token() {
        let tokens = Tokens::from_plaintext("loqui_correct-horse-battery", TokenSource::Environment).unwrap();
        assert!(tokens.accepts_header(Some("Bearer loqui_correct-horse-battery")));
        assert!(!tokens.accepts_header(Some("Bearer loqui_correct-horse-batter")));
        assert!(!tokens.accepts_header(Some("Basic bG9xdWk=")));
        assert!(!tokens.accepts_header(Some("loqui_correct-horse-battery")));
        assert!(!tokens.accepts_header(None));
    }

    #[test]
    fn short_or_unprintable_tokens_are_rejected() {
        assert!(Tokens::from_plaintext("short", TokenSource::Environment).is_err());
        assert!(Tokens::from_plaintext("has a space in it ok", TokenSource::Environment).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn generated_file_is_private_and_reloads() {
        use std::os::unix::fs::PermissionsExt;
        let dir = temp_dir("gen");
        let path = dir.join("token");
        let tokens = Tokens::resolve(None, None, &path).unwrap();
        assert_eq!(tokens.source(), &TokenSource::Generated(path.clone()));
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        let token = std::fs::read_to_string(&path).unwrap();
        assert!(Tokens::resolve(None, None, &path).unwrap().accepts(token.trim()));
        rotate(&path).unwrap();
        assert!(!Tokens::resolve(None, None, &path).unwrap().accepts(token.trim()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn readable_or_linked_token_files_are_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = temp_dir("perm");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("token");
        std::fs::write(&path, "loqui_0123456789abcdef\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let err = Tokens::resolve(Some(&path), None, &dir.join("unused")).unwrap_err().to_string();
        assert!(err.contains("chmod 600"), "{err}");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(Tokens::resolve(Some(&path), None, &dir.join("unused")).is_ok());
        let link = dir.join("link");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(Tokens::resolve(Some(&link), None, &dir.join("unused")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_file_can_hold_several_tokens() {
        let dir = temp_dir("multi");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokens");
        std::fs::write(&path, "# client one\nloqui_aaaaaaaaaaaaaaaa\n\nloqui_bbbbbbbbbbbbbbbb\n").unwrap();
        #[cfg(unix)]
        std::fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o600)).unwrap();
        let tokens = Tokens::resolve(Some(&path), None, &dir.join("unused")).unwrap();
        assert!(tokens.accepts("loqui_aaaaaaaaaaaaaaaa") && tokens.accepts("loqui_bbbbbbbbbbbbbbbb"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
