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
    Unix { listener: UnixListener, path: PathBuf, allowed_uids: Vec<u32> },
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
pub(crate) async fn bind(listen: &Listen, allowed_uids: &[u32]) -> Result<Bound, Error> {
    match listen {
        Listen::Unix(path) => {
            let path = match path {
                Some(p) => p.clone(),
                None => crate::fs::default_socket_path()
                    .ok_or_else(|| Error::Config("XDG_RUNTIME_DIR is not set; pass --listen unix:/path/to/loqui.sock".into()))?,
            };
            let listener = bind_unix(&path)?;
            let mut uids = allowed_uids.to_vec();
            #[cfg(unix)]
            if uids.is_empty() {
                uids.push(crate::fs::current_uid());
            }
            Ok(Bound::Unix { listener, path, allowed_uids: uids })
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
/// whose uid is not allowed) and the loop should carry on.
async fn accept(bound: &Bound) -> std::io::Result<Option<(Io, Transport)>> {
    match bound {
        Bound::Unix { listener, allowed_uids, .. } => {
            let (stream, _) = listener.accept().await?;
            let uid = stream.peer_cred()?.uid();
            if !allowed_uids.contains(&uid) {
                tracing::warn!(uid, "refused a Unix socket connection from a uid that is not allowed");
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
    use rustls_pemfile::{certs, private_key};
    let read = |p: &Path| std::fs::read(p).map_err(|e| Error::Config(format!("{}: {e}", p.display())));
    crate::auth::check_private_file(key).map_err(|e| Error::Config(format!("TLS key: {e}")))?;
    let chain =
        certs(&mut read(cert)?.as_slice()).collect::<Result<Vec<_>, _>>().map_err(|e| Error::Config(format!("{}: {e}", cert.display())))?;
    let key = private_key(&mut read(key)?.as_slice())
        .map_err(|e| Error::Config(format!("{}: {e}", key.display())))?
        .ok_or_else(|| Error::Config(format!("{} holds no private key", key.display())))?;
    let config = tokio_rustls::rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(chain, key)
        .map_err(|e| Error::Config(format!("TLS: {e}")))?;
    Ok(tokio_rustls::TlsAcceptor::from(Arc::new(config)))
}
