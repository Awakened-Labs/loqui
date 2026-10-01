# Security

## Reporting a vulnerability

Report vulnerabilities privately through GitHub:
**[Report a vulnerability](https://github.com/Awakened-Labs/loqui/security/advisories/new)**
(the Security tab of this repository). Please do not open a public issue,
pull request or discussion for a suspected vulnerability.

Include what you can of: the affected crate and version, the configuration
(`loqui serve` flags or the `ServerConfig` you built), a reproduction, and
what an attacker gains. We will acknowledge the report, keep you informed
while we fix it, and credit you in the advisory unless you would rather we
did not.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |

Fixes land on the latest release; there are no backports before 1.0.

## The exposure model

loqui is built so that the easy configuration is the narrow one, and every
widening is a separate, named decision. The rules below are enforced in
code (`crates/loqui-server`, `crates/loqui`); a configuration that breaks
one is refused at startup, before anything is bound.

### The library opens nothing

`loqui::Engine` runs Kokoro and Whisper in process. It opens no socket and
listens on nothing. Its only network access is fetching model weights from
Hugging Face, and `Downloads::Deny` (`--offline`) turns that off.

### Where the server listens

| `--listen` | Reachable by | Token | Transport |
|---|---|---|---|
| `unix` (default) | processes running as this user, on this host | optional (`--require-token`) | Unix socket, mode 0600, in a 0700 directory under `$XDG_RUNTIME_DIR` |
| `loopback:PORT` | anything on this host, any user | required | plaintext is accepted |
| `tcp:ADDR:PORT` | the network of one interface | required | TLS, or `--insecure-plaintext-network` |
| `all:PORT` | every interface; also needs `--i-understand-this-exposes-all-interfaces` | required | TLS, or `--insecure-plaintext-network` |

- On the Unix socket, the peer's uid is read from the kernel
  (`SO_PEERCRED`) for every connection, and only this user is accepted. An
  existing socket path is replaced only if it is a socket owned by this
  user.
- A wildcard address must be spelled `all:PORT`; `tcp:0.0.0.0:…` and
  `tcp:[::]:…` are refused.
- Loopback still needs a token. Other local users can reach 127.0.0.1, and
  so can web pages through DNS rebinding.
- `loqui doctor --listen …` prints what a configuration would expose, and
  whether loqui would accept it, without starting a server.

### Tokens

- Accepted only in an `Authorization: Bearer` header, never in a query
  string, where it would end up in logs and history.
- Only SHA-256 digests of the accepted tokens are held in memory. A
  presented token is compared against each digest in constant time.
  Plaintext is zeroed after it is read.
- Sources, in order:
  1. an explicit token file (one token per line);
  2. `LOQUI_TOKEN`;
  3. a token generated from OS randomness on first run and stored in the
     user's config directory with mode 0600.

  `LOQUI_TOKEN` is discouraged, because other processes of the same user can
  read `/proc/<pid>/environ`.
- A token file must be a regular file (symlinks are refused), owned by the
  current user, with no group or other permissions. Anything else is
  refused rather than trusted. TLS private keys are held to the same rule.

### Requests

The policy layer runs before routing. Its checks run in this order, so a
cross-site page learns nothing from the difference between 401 and 200:

1. **Browsers are refused.** Any request carrying an `Origin` header gets
   403. loqui serves programs, not pages; there is no CORS to configure.
2. **`Host` is checked on TCP.** On loopback it must be `localhost`,
   `127.0.0.1` or `[::1]`; on an interface, that interface's address. With
   `all:PORT` any `Host` is accepted until `--allowed-host` names the
   accepted ones. This defeats DNS rebinding.
3. **Authentication.** On TCP, every route except `/health` on loopback
   needs a token. On a network listener, `/health` needs one too, unless
   `--public-health` is given.

There is deliberately nothing else on the surface. The server has no web
UI, no OpenAPI document and no model management over HTTP: models are
chosen when it starts. A request can only download voice packs on the
built-in roster, never an arbitrary file.

### Limits

| Bound | Default |
|---|---|
| `/v1/audio/speech` body | 64 KiB |
| upload body | 25 MiB (OpenAI's limit) |
| input text | 4096 characters |
| uploaded audio | 1800 s; decoding stops as soon as it is exceeded |
| inference | one at a time per model; 8 more may wait, then 503 with `Retry-After` |
| open connections | 64 |
| request headers | must arrive within 10 s (slow-drip clients are dropped) |
| TLS handshake | 10 s, off the accept loop, so one client cannot stall others |
| request | 120 s |

Error responses use OpenAI's shape. They describe the problem, but never
echo the caller's input or include internal paths.

### Models

- Every download is pinned to a Hugging Face commit revision, so a change
  upstream cannot change what loqui runs.
- The Kokoro and Whisper weights are also checked against pinned SHA-256
  digests before use. Kokoro voice packs (small style vectors) are pinned
  by revision only.
- The cache directory is kept private (0700). The weights are not secret,
  but a directory others can write to would let them substitute a model.
- The G2P lexicons and weights are embedded in the `loqui-g2p` crate and are
  never fetched.

## Scope

In scope, for example:

- reaching the API without a valid token, or from another uid;
- a configuration that exposes more than `loqui doctor` reports, or more
  than the flags given ask for;
- a web page driving the API (CORS, DNS rebinding, `Origin`/`Host` bypass);
- a request that escapes the limits above, or that exhausts memory or CPU
  well beyond them;
- crashes, hangs or memory corruption from crafted audio or text;
- a token, key or path leaking through responses or logs;
- loading weights that do not match their pinned digest.

Out of scope:

- an attacker already running as the same user as loqui. They can read its
  token file and its memory;
- deployments that override a safeguard with its acknowledgement flag, as
  far as the flag's documented consequence goes (for example, plaintext
  tokens crossing the network under `--insecure-plaintext-network`);
- vulnerabilities in dependencies with no loqui-specific impact (please
  report those upstream; we will update);
- the quality of speech or transcription output.

## Known limitations

- Uploaded audio is parsed by symphonia and opus-rs, both pure Rust. They
  are not fuzzed by this project, and a decoder bug is a denial of service.
  Treat the upload limits as part of your defence.
- opus-rs crashes on CPUs with AVX but no FMA (restsend/opus-rs#30), so
  loqui refuses Opus input on those CPUs rather than decoding it.
- With `--require-token` off (the default), the Unix socket trusts every
  process running as this user. That is the boundary it is designed around.
