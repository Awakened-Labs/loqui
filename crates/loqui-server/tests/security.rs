//! The exposure and authentication guarantees, tested against a real bound
//! server over real sockets. No model weights are needed: every request here
//! is decided before inference runs.

use std::path::{Path, PathBuf};
use std::time::Duration;

use loqui_server::{Acknowledgements, Listen, Server, ServerConfig};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const TOKEN: &str = "loqui_test-token-0123456789abcdef";

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("loqui-sec-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn token_file(dir: &Path) -> PathBuf {
    let path = dir.join("token");
    std::fs::write(&path, format!("{TOKEN}\n")).unwrap();
    #[cfg(unix)]
    std::fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o600)).unwrap();
    path
}

fn engine(dir: &Path) -> loqui::Engine {
    loqui::Engine::builder().cache_dir(dir.join("cache")).downloads(loqui::Downloads::Deny).build().unwrap()
}

/// Starts a server; returns its address and a handle that stops it on drop.
async fn start(config: ServerConfig, dir: &Path) -> (String, tokio::sync::oneshot::Sender<()>) {
    let server = Server::bind(config, engine(dir)).await.unwrap();
    let address = server.address();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(server.run(async move {
        let _ = stopped.await;
    }));
    (address, stop)
}

/// Sends one raw HTTP/1.1 request and returns (status, headers+body text).
async fn send(address: &str, request: &str) -> (u16, String) {
    let mut response = Vec::new();
    if let Some(path) = address.strip_prefix("unix:") {
        let mut s = tokio::net::UnixStream::connect(path).await.unwrap();
        // A refused peer sees the connection close or reset; that is an
        // empty response here, not a test failure.
        if s.write_all(request.as_bytes()).await.is_ok() {
            let _ = s.read_to_end(&mut response).await;
        }
    } else {
        let addr = address.strip_prefix("http://").unwrap();
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        s.write_all(request.as_bytes()).await.unwrap();
        s.read_to_end(&mut response).await.unwrap();
    }
    let text = String::from_utf8_lossy(&response).into_owned();
    let status = text.split(' ').nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    (status, text)
}

fn get(path: &str, host: &str, headers: &[&str]) -> String {
    let mut r = format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n");
    for h in headers {
        r.push_str(h);
        r.push_str("\r\n");
    }
    r.push_str("\r\n");
    r
}

fn bearer() -> String {
    format!("Authorization: Bearer {TOKEN}")
}

fn loopback_config(dir: &Path) -> ServerConfig {
    ServerConfig { listen: Listen::Loopback(0), token_file: Some(token_file(dir)), ..ServerConfig::default() }
}

fn host_of(address: &str) -> String {
    address.strip_prefix("http://").unwrap().to_owned()
}

#[tokio::test]
async fn loopback_requires_a_bearer_token() {
    let dir = TempDir::new("loopback-token");
    let (addr, _stop) = start(loopback_config(&dir.0), &dir.0).await;
    let host = host_of(&addr);

    let (status, text) = send(&addr, &get("/v1/models", &host, &[])).await;
    assert_eq!(status, 401, "{text}");
    assert!(text.to_ascii_lowercase().contains("www-authenticate: bearer"), "{text}");

    let (status, _) = send(&addr, &get("/v1/models", &host, &["Authorization: Bearer loqui_wrong-token-0000000000"])).await;
    assert_eq!(status, 401);

    let (status, text) = send(&addr, &get("/v1/models", &host, &[&bearer()])).await;
    assert_eq!(status, 200, "{text}");
    assert!(text.contains("\"owned_by\":\"loqui/tts\""), "{text}");
}

#[tokio::test]
async fn tokens_in_the_query_string_are_ignored() {
    let dir = TempDir::new("query-token");
    let (addr, _stop) = start(loopback_config(&dir.0), &dir.0).await;
    let (status, _) = send(&addr, &get(&format!("/v1/models?api_key={TOKEN}"), &host_of(&addr), &[])).await;
    assert_eq!(status, 401);
}

#[tokio::test]
async fn browser_requests_are_refused_even_with_a_token() {
    let dir = TempDir::new("origin");
    let (addr, _stop) = start(loopback_config(&dir.0), &dir.0).await;
    let (status, text) = send(&addr, &get("/v1/models", &host_of(&addr), &[&bearer(), "Origin: https://evil.example"])).await;
    assert_eq!(status, 403, "{text}");
    assert!(!text.to_ascii_lowercase().contains("access-control-allow-origin"), "no CORS headers: {text}");
}

#[tokio::test]
async fn unexpected_host_headers_are_refused() {
    // DNS rebinding: a page on attacker.example resolves its own name to
    // 127.0.0.1; the request then carries Host: attacker.example.
    let dir = TempDir::new("host");
    let (addr, _stop) = start(loopback_config(&dir.0), &dir.0).await;
    let (status, _) = send(&addr, &get("/v1/models", "attacker.example", &[&bearer()])).await;
    assert_eq!(status, 403);
    let port = host_of(&addr).rsplit_once(':').unwrap().1.to_owned();
    let (status, _) = send(&addr, &get("/v1/models", &format!("localhost:{port}"), &[&bearer()])).await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn health_is_open_on_loopback_and_reveals_nothing() {
    let dir = TempDir::new("health");
    let (addr, _stop) = start(loopback_config(&dir.0), &dir.0).await;
    let (status, text) = send(&addr, &get("/health", &host_of(&addr), &[])).await;
    assert_eq!(status, 200);
    assert!(text.ends_with("{\"status\":\"ok\"}"), "{text}");
}

#[tokio::test]
async fn there_is_no_web_ui_or_api_document() {
    let dir = TempDir::new("surface");
    let (addr, _stop) = start(loopback_config(&dir.0), &dir.0).await;
    for path in ["/", "/web", "/docs", "/openapi.json", "/api/models", "/v1/audio/models/load"] {
        let (status, _) = send(&addr, &get(path, &host_of(&addr), &[&bearer()])).await;
        assert!(status == 404 || status == 405, "{path}: {status}");
    }
}

#[tokio::test]
async fn oversized_speech_bodies_are_rejected() {
    let dir = TempDir::new("body");
    let (addr, _stop) = start(loopback_config(&dir.0), &dir.0).await;
    let body = format!("{{\"model\":\"kokoro\",\"input\":\"{}\"}}", "a".repeat(100 * 1024));
    let request = format!(
        "POST /v1/audio/speech HTTP/1.1\r\nHost: {}\r\n{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        host_of(&addr),
        bearer(),
        body.len()
    );
    let (status, _) = send(&addr, &request).await;
    assert_eq!(status, 413);
}

#[tokio::test]
async fn voice_ids_cannot_reach_the_filesystem() {
    let dir = TempDir::new("voice");
    let (addr, _stop) = start(loopback_config(&dir.0), &dir.0).await;
    for voice in ["../../etc/passwd", "zz_nonexistent"] {
        let body = format!("{{\"model\":\"kokoro\",\"input\":\"hello\",\"voice\":\"{voice}\"}}");
        let request = format!(
            "POST /v1/audio/speech HTTP/1.1\r\nHost: {}\r\n{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            host_of(&addr),
            bearer(),
            body.len()
        );
        let (status, text) = send(&addr, &request).await;
        assert_eq!(status, 400, "{voice}: {text}");
        assert!(!text.contains("etc/passwd"), "errors must not echo input: {text}");
    }
}

#[cfg(unix)]
#[tokio::test]
async fn unix_socket_is_private_and_needs_no_token_for_the_owner() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new("unix");
    let socket = dir.0.join("run").join("loqui.sock");
    let config =
        ServerConfig { listen: Listen::Unix(Some(socket.clone())), token_file: Some(token_file(&dir.0)), ..ServerConfig::default() };
    let (addr, stop) = start(config, &dir.0).await;
    assert_eq!(std::fs::metadata(&socket).unwrap().permissions().mode() & 0o777, 0o600);
    assert_eq!(std::fs::metadata(socket.parent().unwrap()).unwrap().permissions().mode() & 0o777, 0o700);

    let (status, text) = send(&addr, &get("/v1/models", "localhost", &[])).await;
    assert_eq!(status, 200, "{text}");

    drop(stop);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!socket.exists(), "the socket is removed on shutdown");
}

#[cfg(unix)]
#[tokio::test]
async fn unix_socket_can_require_a_token() {
    let dir = TempDir::new("unix-token");
    let socket = dir.0.join("run").join("loqui.sock");
    let config = ServerConfig {
        listen: Listen::Unix(Some(socket)),
        token_file: Some(token_file(&dir.0)),
        require_token_on_unix: true,
        ..ServerConfig::default()
    };
    let (addr, _stop) = start(config, &dir.0).await;
    assert_eq!(send(&addr, &get("/v1/models", "localhost", &[])).await.0, 401);
    assert_eq!(send(&addr, &get("/v1/models", "localhost", &[&bearer()])).await.0, 200);
}

#[cfg(unix)]
#[tokio::test]
async fn unix_socket_refuses_other_uids() {
    let dir = TempDir::new("uid");
    let socket = dir.0.join("run").join("loqui.sock");
    // Allow only a uid that is not ours: our own connection must be dropped.
    let config = ServerConfig {
        listen: Listen::Unix(Some(socket)),
        token_file: Some(token_file(&dir.0)),
        allowed_uids: vec![loqui_server::fs::current_uid().wrapping_add(4242)],
        ..ServerConfig::default()
    };
    let (addr, _stop) = start(config, &dir.0).await;
    let (status, text) = send(&addr, &get("/v1/models", "localhost", &[])).await;
    assert_eq!(status, 0, "the connection is closed without a response: {text:?}");
}

#[cfg(unix)]
#[tokio::test]
async fn a_file_at_the_socket_path_is_never_replaced() {
    let dir = TempDir::new("squat");
    let socket = dir.0.join("run").join("loqui.sock");
    std::fs::create_dir_all(socket.parent().unwrap()).unwrap();
    std::fs::write(&socket, "not a socket").unwrap();
    let config =
        ServerConfig { listen: Listen::Unix(Some(socket.clone())), token_file: Some(token_file(&dir.0)), ..ServerConfig::default() };
    let err = Server::bind(config, engine(&dir.0)).await.err().expect("bind must refuse").to_string();
    assert!(err.contains("refusing"), "{err}");
    assert_eq!(std::fs::read_to_string(&socket).unwrap(), "not a socket");
}

#[tokio::test]
async fn network_listeners_need_explicit_acknowledgement() {
    let dir = TempDir::new("ack");
    let refused = [
        ServerConfig { listen: Listen::AllInterfaces(0), token_file: Some(token_file(&dir.0)), ..ServerConfig::default() },
        ServerConfig {
            listen: Listen::AllInterfaces(0),
            acknowledgements: Acknowledgements { all_interfaces: true, plaintext_network: false },
            token_file: Some(token_file(&dir.0)),
            ..ServerConfig::default()
        },
    ];
    for config in refused {
        assert!(Server::bind(config, engine(&dir.0)).await.is_err());
    }
}
