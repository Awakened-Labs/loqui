//! Binding, accepting and serving connections.
//!
//! loqui runs its own accept loop rather than `axum::serve` because it needs
//! per-connection decisions: the peer's uid on the Unix socket, a TLS
//! handshake that cannot stall other clients, a deadline for request
//! headers (slow-drip clients), and a cap on open connections.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use hyper::body::Incoming;
use hyper_util::rt::{TokioIo, TokioTimer};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, UnixListener};
use tokio::sync::Semaphore;
use tower::ServiceExt;

use crate::Error;
use crate::listen::Listen;
use crate::policy::Transport;

/// Request headers must arrive within this, or the connection is closed.
const HEADER_TIMEOUT: Duration = Duration::from_secs(10);
/// A TLS handshake must finish within this.
#[cfg(feature = "tls")]
const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) enum Bound {
    /// `owner` is the only uid served: the one loqui runs as.
    Unix {
        listener: UnixListener,
        path: PathBuf,
        owner: u32,
    },
    Tcp(TcpListener),
}

impl Bound {
    pub fn describe(&self) -> String {
        match self {
            Self::Unix { path, .. } => format!("unix:{}", path.display()),
            Self::Tcp(l) => l.local_addr().map_or_else(|_| "tcp".into(), |a| format!("http://{a}")),
        }
    }
}

/// Binds the listener `listen` asks for.
pub(crate) async fn bind(listen: &Listen) -> Result<Bound, Error> {
    match listen {
        Listen::Unix(path) => {
            let path = match path {
                Some(p) => p.clone(),
                None => crate::fs::default_socket_path()
                    .ok_or_else(|| Error::Config("XDG_RUNTIME_DIR is not set; pass --listen unix:/path/to/loqui.sock".into()))?,
            };
            let listener = bind_unix(&path)?;
            Ok(Bound::Unix { listener, path, owner: crate::fs::current_uid() })
        }
        tcp => {
            let addr = tcp.socket_addr().expect("TCP modes have an address");
            let listener = TcpListener::bind(addr).await.map_err(|e| Error::Io(format!("binding {addr}: {e}")))?;
            Ok(Bound::Tcp(listener))
        }
    }
}

/// Binds a Unix socket with mode 0600 in a 0700 directory. A leftover
/// socket from a previous run is replaced only if it is a socket we own;
/// anything else at that path is left alone and reported.
fn bind_unix(path: &Path) -> Result<UnixListener, Error> {
    let dir = path.parent().ok_or_else(|| Error::Config(format!("{} has no parent directory", path.display())))?;
    crate::fs::private_dir(dir)?;
    if let Ok(meta) = std::fs::symlink_metadata(path) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::{FileTypeExt, MetadataExt};
            if !meta.file_type().is_socket() || meta.uid() != crate::fs::current_uid() {
                return Err(Error::Config(format!(
                    "{} exists and is not a socket owned by this user; refusing to replace it",
                    path.display()
                )));
            }
        }
        if std::os::unix::net::UnixStream::connect(path).is_ok() {
            return Err(Error::Config(format!("another server is already listening on {}", path.display())));
        }
        std::fs::remove_file(path).map_err(|e| Error::Io(format!("removing stale {}: {e}", path.display())))?;
        let _ = meta;
    }
    let listener = UnixListener::bind(path).map_err(|e| Error::Io(format!("binding {}: {e}", path.display())))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| Error::Io(format!("securing {}: {e}", path.display())))?;
    }
    Ok(listener)
}

#[cfg(feature = "tls")]
pub(crate) type Tls = Option<tokio_rustls::TlsAcceptor>;
#[cfg(not(feature = "tls"))]
pub(crate) type Tls = Option<std::convert::Infallible>;

/// Serves `router` on `bound` until `shutdown` resolves.
pub(crate) async fn run(
    bound: Bound,
    router: Router,
    tls: Tls,
    max_connections: usize,
    shutdown: impl Future<Output = ()>,
) -> Result<(), Error> {
    let connections = Arc::new(Semaphore::new(max_connections));
    tokio::pin!(shutdown);
    let socket_path = match &bound {
        Bound::Unix { path, .. } => Some(path.clone()),
        Bound::Tcp(_) => None,
    };
    let result = loop {
        let permit = tokio::select! {
            () = &mut shutdown => break Ok(()),
            permit = Arc::clone(&connections).acquire_owned() => permit.map_err(|_| Error::Io("connection limiter closed".into()))?,
        };
        let accepted = tokio::select! {
            () = &mut shutdown => break Ok(()),
            accepted = accept(&bound) => accepted,
        };
        let (io, transport) = match accepted {
            Ok(Some(conn)) => conn,
            Ok(None) => continue,
            Err(e) => {
                // Out of file descriptors and similar: back off, keep serving.
                tracing::warn!("accept failed: {e}");
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        let router = router.clone();
        // Without TLS, `Tls` is an empty type and therefore `Copy`.
        #[cfg_attr(not(feature = "tls"), allow(clippy::clone_on_copy))]
        let tls = tls.clone();
        tokio::spawn(async move {
            let _permit = permit;
            match io {
                Io::Unix(s) => serve_connection(s, router, transport).await,
                Io::Tcp(s) => match tls {
                    #[cfg(feature = "tls")]
                    Some(acceptor) => match tokio::time::timeout(TLS_HANDSHAKE_TIMEOUT, acceptor.accept(s)).await {
                        Ok(Ok(stream)) => serve_connection(stream, router, transport).await,
                        Ok(Err(e)) => tracing::debug!("TLS handshake failed: {e}"),
                        Err(_) => tracing::debug!("TLS handshake timed out"),
                    },
                    #[cfg(not(feature = "tls"))]
                    Some(never) => match never {},
                    None => serve_connection(s, router, transport).await,
                },
            }
        });
    };
    if let Some(path) = socket_path {
        let _ = std::fs::remove_file(path);
    }
    result
}

enum Io {
    Unix(tokio::net::UnixStream),
    Tcp(tokio::net::TcpStream),
}

/// Accepts one connection. `Ok(None)` means one was refused (a Unix peer
/// running as another user) and the loop should carry on.
async fn accept(bound: &Bound) -> std::io::Result<Option<(Io, Transport)>> {
    match bound {
        Bound::Unix { listener, owner, .. } => {
            let (stream, _) = listener.accept().await?;
            let uid = stream.peer_cred()?.uid();
            if uid != *owner {
                tracing::warn!(uid, "refused a Unix socket connection from another user");
                return Ok(None);
            }
            Ok(Some((Io::Unix(stream), Transport::Unix { uid })))
        }
        Bound::Tcp(listener) => {
            let (stream, peer) = listener.accept().await?;
            let _ = stream.set_nodelay(true);
            Ok(Some((Io::Tcp(stream), Transport::Tcp { peer })))
        }
    }
}

async fn serve_connection<S>(stream: S, router: Router, transport: Transport)
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let service = hyper::service::service_fn(move |request: hyper::Request<Incoming>| {
        let mut request = request.map(Body::new);
        request.extensions_mut().insert(transport.clone());
        router.clone().oneshot(request)
    });
    let result = hyper::server::conn::http1::Builder::new()
        .timer(TokioTimer::new())
        .header_read_timeout(HEADER_TIMEOUT)
        .keep_alive(true)
        .serve_connection(TokioIo::new(stream), service)
        .await;
    if let Err(e) = result {
        tracing::debug!("connection ended with an error: {e}");
    }
}

#[cfg(feature = "tls")]
pub(crate) fn tls_acceptor(cert: &Path, key: &Path) -> Result<tokio_rustls::TlsAcceptor, Error> {
    use tokio_rustls::rustls::pki_types::pem::{self, PemObject};
    use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer};
    let read = |p: &Path| std::fs::read(p).map_err(|e| Error::Config(format!("{}: {e}", p.display())));
    crate::auth::check_private_file(key).map_err(|e| Error::Config(format!("TLS key: {e}")))?;
    let chain = CertificateDer::pem_slice_iter(&read(cert)?)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| Error::Config(format!("{}: {e}", cert.display())))?;
    let key = PrivateKeyDer::from_pem_slice(&read(key)?).map_err(|e| match e {
        pem::Error::NoItemsFound => Error::Config(format!("{} holds no private key", key.display())),
        e => Error::Config(format!("{}: {e}", key.display())),
    })?;
    let config = tokio_rustls::rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(chain, key)
        .map_err(|e| Error::Config(format!("TLS: {e}")))?;
    Ok(tokio_rustls::TlsAcceptor::from(Arc::new(config)))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// Another user cannot normally reach the 0600 socket at all; if one
    /// does (root, or a loosened mode), the peer-uid check still drops it.
    #[tokio::test]
    async fn a_peer_running_as_another_user_is_dropped() {
        let dir = std::env::temp_dir().join(format!("loqui-peer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("loqui.sock");
        let Bound::Unix { listener, path, .. } = bind(&Listen::Unix(Some(path))).await.unwrap() else { unreachable!() };
        let stranger = Bound::Unix { listener, path: path.clone(), owner: crate::fs::current_uid().wrapping_add(4242) };
        let _client = tokio::net::UnixStream::connect(&path).await.unwrap();
        assert!(accept(&stranger).await.unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(all(test, feature = "tls", unix))]
mod tls_tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    // A throwaway self-signed pair for loqui.test, made for these tests and
    // trusted by nothing.
    const CERT: &str = include_str!("../tests/data/test-cert.pem");
    const KEY: &str = include_str!("../tests/data/test-key.pem");

    /// Writes a certificate and a key (with `key_mode`) and returns their paths.
    fn pair(name: &str, key: &str, key_mode: u32) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("loqui-tls-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (cert_path, key_path) = (dir.join("cert.pem"), dir.join("key.pem"));
        std::fs::write(&cert_path, CERT).unwrap();
        std::fs::write(&key_path, key).unwrap();
        std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(key_mode)).unwrap();
        (cert_path, key_path)
    }

    fn refusal(cert: &Path, key: &Path) -> String {
        match tls_acceptor(cert, key) {
            Ok(_) => panic!("accepted"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn a_certificate_and_its_private_key_load() {
        let (cert, key) = pair("ok", KEY, 0o600);
        assert!(tls_acceptor(&cert, &key).is_ok());
    }

    #[test]
    fn a_key_file_without_a_key_is_named() {
        let (cert, key) = pair("nokey", CERT, 0o600);
        assert!(refusal(&cert, &key).contains("holds no private key"));
    }

    #[test]
    fn a_key_others_can_read_is_refused() {
        let (cert, key) = pair("mode", KEY, 0o640);
        assert!(refusal(&cert, &key).contains("chmod 600"));
    }

    #[test]
    fn a_key_that_does_not_match_the_certificate_is_refused() {
        // A valid P-256 key, but not the one the certificate was issued for.
        const OTHER: &str = "-----BEGIN PRIVATE KEY-----\n\
            MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgevZzL1gdAFr88hb2\n\
            OF/2NxApJCzGCEDdfSp6VQO30hyhRANCAAQRWz+jn65BtOMvdyHKcvjBeBSDZH2r\n\
            1RTwjmYSi9R/zpBnuQ4EiMnCqfMPWiZqB4QdbAd0E7oH50VpuZ1P087G\n\
            -----END PRIVATE KEY-----\n";
        let (cert, key) = pair("mismatch", OTHER, 0o600);
        assert!(refusal(&cert, &key).contains("KeyMismatch"));
    }
}
