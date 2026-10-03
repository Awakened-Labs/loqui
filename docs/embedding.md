# Embedding loqui safely

loqui can run inside your program in two ways: as a library
(`loqui::Engine`, with no listener at all), or as a server you start
yourself (`loqui_server::Server`). Either way, the defaults are the narrow
choices. This guide covers what each default protects, what it costs to
change one, and the licensing a program takes on with each feature.

Both examples below compile with the workspace, so they stay accurate:
`crates/loqui/examples/embed.rs` and
`crates/loqui-server/examples/embed_server.rs`.

## In process: `loqui::Engine`

```rust
let builder = loqui::Engine::builder()
    .downloads(loqui::Downloads::Deny)   // never touch the network
    .max_input_chars(1000)               // default 4096
    .max_audio_secs(120);                // default 1800
#[cfg(feature = "whisper")]
let builder = builder.stt(Some(loqui::SttConfig::default()));   // STT is opt-in
let engine = builder.build()?;

let speech = engine.speak(&loqui::SpeakRequest::new("Hello."))?;
```

**There is no listener.** An `Engine` opens no socket. Anything that can
call it is already inside your process, so the protections that matter are
about what your callers can make it do.

**Downloads.** By default, missing weights are fetched from Hugging Face,
pinned to a commit revision. The Kokoro and Whisper weights are also checked
against a SHA-256 digest. A service that should never reach the network
should use `Downloads::Deny` and fill the cache ahead of time: with
`loqui fetch`, or by calling `Engine::fetch()` (and `fetch_voices()` for
every voice) where downloads are allowed. `fetch()` verifies each digest but
loads nothing. `Engine::missing()` lists what is still absent, with sizes,
without loading, hashing or downloading anything, so a health check can call
it. Shipping a read-only cache, such as a container layer or a mounted
volume, works: a cache directory that is already private is left as it is.

**The cache directory** (`$XDG_CACHE_HOME/loqui` by default, or
`cache_dir(..)`) is created mode 0700. The weights are not secret, but
anyone who can write there can substitute a model. If you choose your own
directory, keep it owned by the service's user and closed to everyone else.

**Untrusted input.** Treat text and audio from your users the way the
server does:

- Text is capped by `max_input_chars`, and audio by `max_audio_secs`.
  Decoding stops as soon as audio passes the cap, so a long file is never
  decoded in full.
- The engine does not limit the size of the bytes you hand it. The server
  caps speech bodies at 64 KiB and uploads at 25 MiB before they reach the
  engine. Do the same where your program receives them.
- Voices are checked against the built-in roster before anything is
  fetched, so a voice name can never name an arbitrary file.
- Uploaded audio is parsed by pure-Rust decoders (symphonia and opus-rs).
  Treat a decoder bug as a possible denial of service, and keep the caps.

**Older CPUs.** Opus is refused, in both directions, on CPUs with AVX but no
FMA (Sandy and Ivy Bridge, some VMs), because opus-rs crashes there. A
program that patches opus-rs with restsend/opus-rs#31, through
`[patch.crates-io]`, can enable `unguarded-opus` to lift the refusal.
Without the patch, that feature turns the error back into a crash.

**Named voices.** `EngineBuilder::voice("will", "am_puck(1)+am_liam(1)")`
names a blend of built-in voices; afterwards `"will"` works anywhere a
voice does, inside other blends too. `build()` checks every name and spec
and returns `Error::Config` for a bad one, so a mistake surfaces at
startup. `Engine::voices()` lists the built-in and named voices, and
`preload(true)` also fetches the packs named voices need, so a cache
prepared for `Downloads::Deny` is checked at startup too (`loqui fetch`
fetches every pack). If you use `loqui-kokoro` directly, a voice is a
`loqui_kokoro::Blend`: `kokoro.speak(text, &"af_bella(2)+af_sky(1)".parse()?, 1.0)`.

**Concurrency.** `Engine` is cheap to clone, and the clones share the loaded
models. Every call is synchronous and CPU- or GPU-bound, from tens of
milliseconds to several seconds, so in async code run it under
`tokio::task::spawn_blocking` (or your runtime's equivalent), never on the
executor. Each model runs one inference at a time. If your callers can
queue work, bound that queue: the server allows 8 waiting requests per model,
then answers 503. `TtsConfig::threads` and `TranscribeRequest::threads` cap the
CPU threads each model uses; by default the runtimes choose.

**Memory.** Kokoro is about 330 MB and stays resident unless you give
`TtsConfig::idle_ttl`. Whisper large-v3-turbo is about 1.6 GB, and by
default it is freed after ten idle minutes (`SttConfig::idle_ttl`). A model
is never unloaded while a request holds it. `preload(true)` loads everything
at `build()`, so the first request does not pay for it.

**GPUs.** The `cuda` feature runs Kokoro through ONNX Runtime's CUDA
provider and Whisper through whisper.cpp's CUDA backend. It needs CUDA 13
and the toolkit at build time. Select the device in `TtsConfig::device` and
`SttConfig::device`. `whisper-cuda` puts only Whisper on the GPU, leaving
Kokoro on ONNX Runtime's CPU build.

## Licensing for embedders

The default dependency graph is permissive (MIT, Apache-2.0, BSD,
Unlicense), apart from one weak-copyleft crate, and links no GPL code:

| Component | License | When |
|---|---|---|
| loqui crates | MIT | always |
| symphonia (audio demuxing and decoding) | MPL-2.0: copyleft per file, used unmodified | always |
| ONNX Runtime | MIT, downloaded prebuilt by `ort` at build time | always (text-to-speech) |
| whisper.cpp | MIT, built from source (cmake, C++) | `whisper` |
| opus-rs | BSD-3-Clause | always |
| LAME | **LGPL**, statically linked | **`mp3` only** |

- **Leave `mp3` off in a proprietary program.** LAME is LGPL and linked
  statically, so a binary that contains it must let its users relink it
  against a modified LAME. Ask for `Format::Opus` or `Format::Wav` instead.
  If you need MP3, run `loqui serve` built with `mp3` as a separate process:
  the obligation then stays with that binary, not with its clients.
- `Format` is `#[non_exhaustive]` because `Mp3` exists only with the
  feature, and any crate in your build can enable it. Keep a wildcard arm
  when you match on it.
- `ort` downloads ONNX Runtime while building. For builds that must not
  reach the network, or distributions that package ONNX Runtime, enable
  `loqui-kokoro`'s `load-dynamic` feature and provide `libonnxruntime` at
  run time.
- Model weights are not redistributed by loqui. Kokoro-82M is Apache-2.0
  and Whisper is MIT. `NOTICE` lists every attribution.

## Embedding the server: `loqui_server::Server`

```rust
let engine = loqui::Engine::builder().downloads(loqui::Downloads::Deny).build()?;
let config = ServerConfig { require_token_on_unix: true, ..ServerConfig::default() };
let server = Server::bind(config, engine).await?;    // refuses unsafe configs before binding
server.run(async { let _ = tokio::signal::ctrl_c().await; }).await?;
```

`Server::bind` checks the whole configuration before it binds anything. It
returns an error rather than starting something wider than you asked for.
Keep that property: surface the error, and do not retry with a looser
configuration.

### Choosing where to listen

Pick the first of these that works for your clients:

1. **The default Unix socket.** Only processes running as the same user can
   connect: the socket is 0600 in a 0700 directory, and each peer's uid is
   checked again on accept. Set `require_token_on_unix` as well if other
   software runs as that user and should not have access.
2. **`Listen::Loopback(port)`** when a client cannot use a Unix socket.
   Every request needs a token, because other local users can reach
   127.0.0.1, and so can web pages through DNS rebinding.
3. **`Listen::Interface(addr)`** for one network, with TLS (`tls_cert` and
   `tls_key`, from a build with the `tls` feature). Without TLS the token
   crosses the network in the clear. Setting
   `acknowledgements.plaintext_network` is right only behind a
   TLS-terminating proxy, or on a network you trust as much as the host.
4. **`Listen::AllInterfaces(port)`** only when you mean every interface.
   It needs `acknowledgements.all_interfaces`, and it should come with
   `allowed_hosts` naming the hosts clients will use; otherwise any `Host`
   header is accepted.

`loqui doctor --listen …` prints what a configuration exposes, and whether
loqui would start with it. Run it against the configuration you plan to
ship.

### Tokens

- Prefer `token_file`. It must be a regular file owned by your service's
  user, with no group or other permissions, and it holds one token per
  line, so you can rotate by adding the new token before removing the old.
- With neither a token file nor `LOQUI_TOKEN`, loqui generates a token and
  stores it in `~/.config/loqui/token` (0600).
- Avoid `LOQUI_TOKEN`: other processes of the same user can read a
  process's environment.
- Clients send `Authorization: Bearer <token>`. A token in a query string is
  never accepted.

### Limits and browsers

- Keep `max_queue` (8), `max_connections` (64) and `request_timeout`
  (120 s) unless you have measured a reason to change them. Raising them
  trades memory and latency for throughput, and a GPU still runs one
  inference at a time per model.
- Body caps (64 KiB of speech JSON, 25 MiB of upload), the 10 s header
  deadline and the 10 s TLS handshake deadline are fixed.
- Requests that carry an `Origin` header are refused, and there is no CORS.
  If a web page must use speech, put your own authenticated backend in
  between; do not expose loqui to the browser.

### What not to add

The server deliberately has no web UI, no OpenAPI page and no model
management over HTTP: models are chosen when the server starts. If you put
routes of your own next to loqui's (behind the same listener or a proxy),
they do not inherit its policy layer. Give them their own authentication,
`Origin` refusal and limits.

See [SECURITY.md](../SECURITY.md) for the full exposure model, and for how
to report a vulnerability.
