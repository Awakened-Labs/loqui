//! Who may call what: authentication, browser refusal, Host checking.
//!
//! Applied to every request before routing. Order matters: browser and Host
//! checks run before authentication, so a cross-site page learns nothing
//! from the difference between a 401 and a 200.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::header;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::auth::Tokens;
use crate::error::ApiError;
use crate::listen::Listen;

/// How a request reached the server, attached per connection.
#[derive(Debug, Clone)]
pub enum Transport {
    /// A Unix socket peer, already checked against the allowed uids.
    Unix { uid: u32 },
    Tcp { peer: SocketAddr },
}

pub struct Policy {
    pub tokens: Tokens,
    pub listen: Listen,
    /// Require a token on the Unix socket too.
    pub require_token_on_unix: bool,
    /// Extra Host names a TCP client may use (e.g. a DNS name for a LAN IP).
    pub allowed_hosts: Vec<String>,
    /// Serve `/health` without a token on network listeners.
    pub public_health: bool,
}

impl Policy {
    fn token_required(&self, transport: &Transport, path: &str) -> bool {
        let local = match transport {
            Transport::Unix { .. } => {
                if !self.require_token_on_unix {
                    return false;
                }
                true
            }
            Transport::Tcp { peer } => peer.ip().is_loopback() && !self.listen.is_network(),
        };
        if path == "/health" && (local || self.public_health) {
            return false;
        }
        true
    }

    /// Host names a TCP request may carry. `None` means any.
    fn host_allowed(&self, host: &str) -> bool {
        let name = strip_port(host).to_ascii_lowercase();
        if self.allowed_hosts.iter().any(|h| h.eq_ignore_ascii_case(&name)) {
            return true;
        }
        match &self.listen {
            Listen::Unix(_) => true,
            Listen::Loopback(_) => matches!(name.as_str(), "localhost" | "127.0.0.1" | "[::1]" | "::1"),
            Listen::Interface(addr) => {
                let ip = addr.ip().to_string();
                name == ip || name == format!("[{ip}]")
            }
            // Every address of the host is valid; restrict with --allowed-host.
            Listen::AllInterfaces(_) => self.allowed_hosts.is_empty(),
        }
    }
}

fn strip_port(host: &str) -> &str {
    if let Some(rest) = host.strip_prefix('[') {
        // [v6]:port
        return rest.split(']').next().map_or(host, |ip| &host[..ip.len() + 2]);
    }
    match host.rsplit_once(':') {
        Some((name, port)) if port.bytes().all(|b| b.is_ascii_digit()) => name,
        _ => host,
    }
}

pub async fn enforce(State(policy): State<Arc<Policy>>, request: Request, next: Next) -> Response {
    let transport = request.extensions().get::<Transport>().cloned();
    let Some(transport) = transport else {
        return ApiError::internal("connection metadata missing").into_response();
    };

    // Browsers attach Origin to cross-site requests and to every POST. loqui
    // serves programs, not pages, so any request that carries one is refused
    // outright; there is no CORS to configure.
    if request.headers().contains_key(header::ORIGIN) {
        return ApiError::forbidden("browser_request", "requests from web pages are not accepted").into_response();
    }

    if matches!(transport, Transport::Tcp { .. }) {
        let host = request.headers().get(header::HOST).and_then(|h| h.to_str().ok());
        if !host.is_some_and(|h| policy.host_allowed(h)) {
            return ApiError::forbidden("host_not_allowed", "this Host is not served here").into_response();
        }
    }

    if policy.token_required(&transport, request.uri().path()) {
        let header = request.headers().get(header::AUTHORIZATION).and_then(|h| h.to_str().ok());
        if !policy.tokens.accepts_header(header) {
            return ApiError::unauthorized().into_response();
        }
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_ports_are_stripped() {
        assert_eq!(strip_port("localhost:8100"), "localhost");
        assert_eq!(strip_port("[::1]:8100"), "[::1]");
        assert_eq!(strip_port("[::1]"), "[::1]");
        assert_eq!(strip_port("example.com"), "example.com");
    }
}
