//! loqui-server inside another program, keeping its defaults: a Unix socket
//! only this user can open, plus a token on it as well, and no downloads at
//! run time.
//!
//!     loqui fetch                                # once, where the network is allowed
//!     cargo run -p loqui-server --example embed_server
//!     curl --unix-socket "$XDG_RUNTIME_DIR/loqui/loqui.sock" \
//!         -H "Authorization: Bearer $(cat ~/.config/loqui/token)" http://localhost/v1/models

use loqui_server::{Server, ServerConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = loqui::Engine::builder().downloads(loqui::Downloads::Deny).build()?;

    // Everything not named here keeps its narrow default: the Unix socket in
    // $XDG_RUNTIME_DIR, the generated token, 8 queued requests per model,
    // 64 connections, 120 s per request.
    let config = ServerConfig { require_token_on_unix: true, ..ServerConfig::default() };

    // Refuses an unsafe configuration before binding anything.
    let server = Server::bind(config, engine).await?;
    eprintln!("serving on {}", server.address());
    server
        .run(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
