//! An authenticated, explicitly exposed HTTP front end for a loqui
//! [`Engine`](loqui::Engine), speaking the OpenAI audio API.
//!
//! The defaults are the narrow ones:
//!
//! - It listens on a Unix socket in `$XDG_RUNTIME_DIR` (mode 0600, in a
//!   0700 directory) and accepts only processes running as the same user.
//!   TCP must be asked for, and network-wide TCP must be asked for twice.
//!   See [`Listen`].
//! - Every TCP request needs a bearer token, loopback included. With none
//!   configured, one is generated and stored with mode 0600. See [`auth`].
//! - Requests from web pages (anything carrying `Origin`) are refused, TCP
//!   requests must name an expected `Host`, and there is no CORS.
//! - Bodies, text length, audio length, queue depth, header arrival and
//!   request time are all bounded.
//!
//! [SECURITY.md](https://github.com/Awakened-Labs/loqui/blob/main/SECURITY.md) describes the exposure model in full, and
//! [Embedding loqui safely](https://github.com/Awakened-Labs/loqui/blob/main/docs/embedding.md) covers choosing a listener and tokens.

pub mod api;
pub mod auth;
mod error;
pub mod fs;
pub mod listen;
pub mod policy;
mod serve;

use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

pub use listen::{Acknowledgements, Listen};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Config(String),
    #[error("token: {0}")]
    Token(String),
    #[error("{0}")]
    Io(String),
    #[error(transparent)]
    Engine(#[from] loqui::Error),
}

/// Everything about how the server is exposed.
#[derive(Debug, Clone, Default)]
pub struct ServerConfig {
    pub listen: Listen,
    pub acknowledgements: Acknowledgements,
    /// Read tokens from this file (one per line). Otherwise `LOQUI_TOKEN`,
    /// otherwise a generated token at [`fs::default_token_path`].
    pub token_file: Option<PathBuf>,
    /// Require a token on the Unix socket as well.
    pub require_token_on_unix: bool,
    /// Extra `Host` names TCP clients may use.
    pub allowed_hosts: Vec<String>,
    pub public_health: bool,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
    /// Requests allowed to wait per model before 503. Default 8.
    pub max_queue: Option<usize>,
    /// Open connections. Default 64.
    pub max_connections: Option<usize>,
    /// Per-request deadline. Default 120 s.
    pub request_timeout: Option<Duration>,
}

/// A server ready to run: bound, with its tokens resolved.
pub struct Server {
    bound: serve::Bound,
    router: axum::Router,
    tls: serve::Tls,
    max_connections: usize,
    token_source: auth::TokenSource,
    listen: Listen,
}

impl Server {
    /// Validates the exposure, resolves tokens and binds. Fails before
    /// binding anything if the configuration is unsafe.
    pub async fn bind(config: ServerConfig, engine: loqui::Engine) -> Result<Self, Error> {
        let tls_requested = config.tls_cert.is_some() || config.tls_key.is_some();
        config.listen.check(config.acknowledgements, tls_requested)?;
        let tls = tls_from(&config)?;

        let env_token = std::env::var("LOQUI_TOKEN").ok().filter(|t| !t.is_empty()).map(zeroize::Zeroizing::new);
        let generated = fs::default_token_path()
            .ok_or_else(|| Error::Config("cannot find a config directory for the token; set HOME or pass --token-file".into()))?;
        let tokens = auth::Tokens::resolve(config.token_file.as_deref(), env_token, &generated)?;
        let token_source = tokens.source().clone();

        let state = Arc::new(api::AppState {
            engine,
            tts_gate: api::Gate::new(config.max_queue.unwrap_or(8)),
            stt_gate: api::Gate::new(config.max_queue.unwrap_or(8)),
            request_timeout: config.request_timeout.unwrap_or(Duration::from_secs(120)),
        });
        let policy = Arc::new(policy::Policy {
            tokens,
            listen: config.listen.clone(),
            require_token_on_unix: config.require_token_on_unix,
            allowed_hosts: config.allowed_hosts.clone(),
            public_health: config.public_health,
        });
        let router = api::router(state).layer(axum::middleware::from_fn_with_state(policy, policy::enforce));
        let bound = serve::bind(&config.listen).await?;
        Ok(Self { bound, router, tls, max_connections: config.max_connections.unwrap_or(64), token_source, listen: config.listen })
    }

    /// Where the server is listening, e.g. `unix:/run/user/1000/loqui/loqui.sock`.
    pub fn address(&self) -> String {
        self.bound.describe()
    }

    pub fn token_source(&self) -> &auth::TokenSource {
        &self.token_source
    }

    pub fn listen(&self) -> &Listen {
        &self.listen
    }

    /// Serves until `shutdown` resolves, then removes the socket file.
    pub async fn run(self, shutdown: impl Future<Output = ()>) -> Result<(), Error> {
        serve::run(self.bound, self.router, self.tls, self.max_connections, shutdown).await
    }
}

#[cfg(feature = "tls")]
fn tls_from(config: &ServerConfig) -> Result<serve::Tls, Error> {
    match (&config.tls_cert, &config.tls_key) {
        (Some(cert), Some(key)) => Ok(Some(serve::tls_acceptor(cert, key)?)),
        (None, None) => Ok(None),
        _ => Err(Error::Config("--tls-cert and --tls-key go together".into())),
    }
}

#[cfg(not(feature = "tls"))]
fn tls_from(config: &ServerConfig) -> Result<serve::Tls, Error> {
    if config.tls_cert.is_some() || config.tls_key.is_some() {
        return Err(Error::Config("TLS was requested but loqui-server was built without the `tls` feature".into()));
    }
    Ok(None)
}
