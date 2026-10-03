//! `loqui`: local speech, served safely.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use loqui::{Downloads, Engine, KokoroVariant, TtsConfig};
use loqui_server::{Acknowledgements, Listen, Server, ServerConfig, auth, fs};

#[derive(Parser)]
#[command(name = "loqui", version, about = "Local Kokoro text-to-speech and Whisper speech-to-text, served safely.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the OpenAI audio API. Listens on a private Unix socket unless
    /// told otherwise.
    Serve(ServeArgs),
    /// Speak text to an audio file, in process.
    Speak(SpeakArgs),
    /// Transcribe an audio file, in process.
    #[cfg(feature = "whisper")]
    Transcribe(TranscribeArgs),
    /// Show, locate or replace the generated API token.
    Token {
        #[command(subcommand)]
        action: TokenAction,
    },
    /// Explain what `serve` would expose with these options, and flag risks.
    Doctor(ServeArgs),
    /// Download model weights and every voice now, for hosts that will run
    /// with --offline.
    Fetch(EngineArgs),
    /// List the voices `speak` and `serve` accept, named voices included.
    Voices(VoicesFile),
}

#[derive(Subcommand)]
enum TokenAction {
    /// Print the token (to stdout, for piping into a client's config).
    Show,
    /// Print where the token file is.
    Path,
    /// Replace the token. Clients using the old one stop working.
    Rotate,
}

#[derive(Args, Clone)]
struct VoicesFile {
    /// Named voices: a TOML file of `name = "blend"` lines, such as
    /// `will = "am_puck(1)+am_liam(1)+am_onyx(0.5)"` [default:
    /// $XDG_CONFIG_HOME/loqui/voices.toml, if it exists].
    #[arg(long = "voices", env = "LOQUI_VOICES_FILE")]
    path: Option<PathBuf>,
}

impl VoicesFile {
    /// The file to read, if any.
    fn path(&self) -> Option<PathBuf> {
        voices_path(self.path.as_deref(), fs::config_dir().as_deref())
    }

    /// Its `(name, spec)` pairs; none without a file.
    fn read(&self) -> Result<Vec<(String, String)>, String> {
        self.path().map_or(Ok(Vec::new()), |path| read_voices(&path))
    }
}

/// The voices file: the one named, else `voices.toml` in the config
/// directory when there is one.
fn voices_path(explicit: Option<&Path>, config_dir: Option<&Path>) -> Option<PathBuf> {
    explicit.map(Path::to_path_buf).or_else(|| config_dir.map(|d| d.join("voices.toml")).filter(|p| p.exists()))
}

/// Reads `name = "spec"` pairs. Names and specs are checked when the engine
/// is built; this only reads them.
fn read_voices(path: &Path) -> Result<Vec<(String, String)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let table: toml::Table = text.parse().map_err(|e| format!("{}: {e}", path.display()))?;
    table
        .into_iter()
        .map(|(name, value)| match value {
            toml::Value::String(spec) => Ok((name, spec)),
            _ => Err(format!("{}: voice {name:?} must be a string such as \"af_bella(2)+af_sky(1)\"", path.display())),
        })
        .collect()
}

#[derive(Args, Clone)]
struct EngineArgs {
    /// Where model weights are cached [default: $XDG_CACHE_HOME/loqui].
    #[arg(long, env = "LOQUI_CACHE_DIR")]
    cache_dir: Option<PathBuf>,
    /// Never download; use only cached weights.
    #[arg(long)]
    offline: bool,
    /// Kokoro precision: fp32 (reference), fp16, or quantized.
    #[arg(long, default_value = "fp32", value_parser = parse_variant)]
    kokoro: KokoroVariant,
    /// Run models on a GPU: cuda or cuda:N. Needs a `cuda` build.
    #[arg(long)]
    gpu: Option<String>,
    /// Whisper model for transcription (large-v3-turbo, small.en, ...).
    #[cfg(feature = "whisper")]
    #[arg(long, default_value = "large-v3-turbo")]
    stt_model: String,
    /// Disable transcription.
    #[cfg(feature = "whisper")]
    #[arg(long)]
    no_stt: bool,
    /// Unload Whisper after this many idle seconds (0 keeps it loaded).
    #[cfg(feature = "whisper")]
    #[arg(long, default_value_t = 600)]
    stt_idle_secs: u64,
    /// Load models at startup rather than on first request.
    #[arg(long)]
    preload: bool,
    #[command(flatten)]
    voices: VoicesFile,
}

#[derive(Args, Clone)]
struct ServeArgs {
    /// unix[:PATH] (default), loopback:PORT, tcp:ADDRESS:PORT or all:PORT.
    #[arg(long, env = "LOQUI_LISTEN", default_value = "unix")]
    listen: String,
    /// Required with --listen all:PORT.
    #[arg(long)]
    i_understand_this_exposes_all_interfaces: bool,
    /// Allow a non-loopback listener without TLS (trusted LAN, VPN, or a TLS
    /// proxy in front).
    #[arg(long)]
    insecure_plaintext_network: bool,
    /// Tokens to accept, one per line [default: LOQUI_TOKEN, else a
    /// generated token].
    #[arg(long, env = "LOQUI_TOKEN_FILE")]
    token_file: Option<PathBuf>,
    /// Require a token on the Unix socket too.
    #[arg(long)]
    require_token: bool,
    /// Also accept these Host names (e.g. a DNS name for a LAN address).
    #[arg(long = "allowed-host")]
    allowed_hosts: Vec<String>,
    /// Serve /health without a token on network listeners.
    #[arg(long)]
    public_health: bool,
    #[arg(long, requires = "tls_key")]
    tls_cert: Option<PathBuf>,
    #[arg(long, requires = "tls_cert")]
    tls_key: Option<PathBuf>,
    /// Longest input text, in characters.
    #[arg(long, default_value_t = 4096)]
    max_input_chars: usize,
    /// Longest uploaded audio, in seconds.
    #[arg(long, default_value_t = 1800)]
    max_audio_secs: u64,
    /// Requests that may wait per model before the server answers 503.
    #[arg(long, default_value_t = 8)]
    max_queue: usize,
    #[command(flatten)]
    engine: EngineArgs,
}

#[derive(Args)]
struct SpeakArgs {
    text: String,
    #[arg(short, long, default_value = "speech.wav")]
    output: PathBuf,
    #[arg(long, default_value = "af_heart")]
    voice: String,
    #[arg(long, default_value_t = 1.0)]
    speed: f32,
    /// wav, flac, opus or pcm; mp3 in builds with the `mp3` feature.
    #[arg(long, default_value = "wav")]
    format: String,
    #[command(flatten)]
    engine: EngineArgs,
}

#[cfg(feature = "whisper")]
#[derive(Args)]
struct TranscribeArgs {
    file: PathBuf,
    #[arg(long)]
    language: Option<String>,
    #[command(flatten)]
    engine: EngineArgs,
}

fn parse_variant(s: &str) -> Result<KokoroVariant, String> {
    match s {
        "fp32" => Ok(KokoroVariant::Fp32),
        "fp16" => Ok(KokoroVariant::Fp16),
        "quantized" | "q8" => Ok(KokoroVariant::Quantized),
        other => Err(format!("unknown Kokoro variant {other:?}; use fp32, fp16 or quantized")),
    }
}

fn gpu_ordinal(gpu: &Option<String>) -> Result<Option<i32>, String> {
    match gpu.as_deref() {
        None => Ok(None),
        Some("cuda") => Ok(Some(0)),
        Some(s) => s.strip_prefix("cuda:").and_then(|n| n.parse().ok()).map(Some).ok_or_else(|| format!("--gpu {s:?}: use cuda or cuda:N")),
    }
}

fn build_engine(args: &EngineArgs, want_stt: bool, max_input_chars: usize, max_audio_secs: u64) -> Result<Engine, String> {
    let gpu = gpu_ordinal(&args.gpu)?;
    let mut builder = Engine::builder()
        .downloads(if args.offline { Downloads::Deny } else { Downloads::Allow })
        .max_input_chars(max_input_chars)
        .max_audio_secs(max_audio_secs)
        .preload(args.preload)
        .tts(Some(TtsConfig {
            variant: args.kokoro,
            device: gpu.map_or(loqui_kokoro::Device::Cpu, loqui_kokoro::Device::Cuda),
            ..TtsConfig::default()
        }));
    if let Some(dir) = &args.cache_dir {
        builder = builder.cache_dir(dir);
    }
    for (name, spec) in args.voices.read()? {
        builder = builder.voice(name, spec);
    }
    #[cfg(feature = "whisper")]
    if want_stt && !args.no_stt {
        builder = builder.stt(Some(loqui::SttConfig {
            model: args.stt_model.clone(),
            device: gpu.map_or(loqui::SttDevice::Cpu, loqui::SttDevice::Gpu),
            idle_ttl: (args.stt_idle_secs > 0).then(|| std::time::Duration::from_secs(args.stt_idle_secs)),
            ..loqui::SttConfig::default()
        }));
    }
    #[cfg(not(feature = "whisper"))]
    let _ = want_stt;
    builder.build().map_err(|e| e.to_string())
}

fn server_config(args: &ServeArgs) -> Result<ServerConfig, String> {
    Ok(ServerConfig {
        listen: args.listen.parse::<Listen>().map_err(|e| e.to_string())?,
        acknowledgements: Acknowledgements {
            all_interfaces: args.i_understand_this_exposes_all_interfaces,
            plaintext_network: args.insecure_plaintext_network,
        },
        token_file: args.token_file.clone(),
        require_token_on_unix: args.require_token,
        allowed_hosts: args.allowed_hosts.clone(),
        public_health: args.public_health,
        tls_cert: args.tls_cert.clone(),
        tls_key: args.tls_key.clone(),
        max_queue: Some(args.max_queue),
        ..ServerConfig::default()
    })
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = term => {}
    }
    tracing::info!("shutting down");
}

async fn serve(args: ServeArgs) -> Result<(), String> {
    let config = server_config(&args)?;
    let engine = build_engine(&args.engine, true, args.max_input_chars, args.max_audio_secs)?;
    let server = Server::bind(config, engine).await.map_err(|e| e.to_string())?;
    tracing::info!(address = %server.address(), token = %server.token_source(), "loqui listening");
    if server.listen().is_network() && args.insecure_plaintext_network && args.tls_cert.is_none() {
        tracing::warn!("serving a network address without TLS; tokens and audio cross the network in plaintext");
    }
    server.run(shutdown_signal()).await.map_err(|e| e.to_string())
}

fn doctor(args: &ServeArgs) -> Result<(), String> {
    let config = server_config(args)?;
    let mut problems = 0;
    let mut say = |ok: bool, line: String| {
        println!("{} {line}", if ok { "ok  " } else { "WARN" });
        if !ok {
            problems += 1;
        }
    };
    let tls = config.tls_cert.is_some();
    match config.listen.check(config.acknowledgements, tls) {
        Ok(()) => say(true, format!("listener {} is allowed", config.listen)),
        Err(e) => say(false, format!("serve would refuse to start: {e}")),
    }
    let reach = match &config.listen {
        Listen::Unix(_) => "processes running as this user, on this host".to_owned(),
        Listen::Loopback(p) => format!("any process of any user on this host, via 127.0.0.1:{p}, with a token"),
        Listen::Interface(a) => format!("anything that can route to {a}, with a token"),
        Listen::AllInterfaces(p) => format!("anything that can reach any address of this host on port {p}, with a token"),
    };
    say(!config.listen.is_network(), format!("reachable by: {reach}"));
    if config.listen.is_unix() {
        match config.listen {
            Listen::Unix(Some(ref p)) => say(true, format!("socket: {}", p.display())),
            _ => match fs::default_socket_path() {
                Some(p) => say(true, format!("socket: {}", p.display())),
                None => say(false, "XDG_RUNTIME_DIR is unset: pass --listen unix:/path".into()),
            },
        }
        say(
            true,
            format!(
                "token on the socket: {}",
                if config.require_token_on_unix { "required" } else { "not required (peer uid is checked)" }
            ),
        );
    }
    if std::env::var_os("LOQUI_TOKEN").is_some() {
        say(false, "LOQUI_TOKEN is set; other processes of this user can read it from /proc. Prefer --token-file".into());
    }
    match (&config.token_file, fs::default_token_path()) {
        (Some(p), _) => match auth::check_private_file(p) {
            Ok(()) => say(true, format!("token file {} is private", p.display())),
            Err(e) => say(false, e.to_string()),
        },
        (None, Some(p)) if p.exists() => match auth::check_private_file(&p) {
            Ok(()) => say(true, format!("generated token {} is private", p.display())),
            Err(e) => say(false, e.to_string()),
        },
        (None, Some(p)) => say(true, format!("a token will be generated at {}", p.display())),
        (None, None) => say(false, "no config directory for a generated token: set HOME or pass --token-file".into()),
    }
    if config.public_health && config.listen.is_network() {
        say(false, "/health is public on a network listener (reveals only that loqui is running)".into());
    }
    let cache = args.engine.cache_dir.clone().or_else(loqui::default_cache_dir);
    say(cache.is_some(), format!("model cache: {}", cache.map_or("none".into(), |c| c.display().to_string())));
    // The engine `serve` would build: a configuration it would refuse, such
    // as a bad voices file, stops it, so say so here.
    match build_engine(&EngineArgs { preload: false, ..args.engine.clone() }, true, 4096, 0) {
        Ok(engine) => {
            if let Some(path) = args.engine.voices.path() {
                let named = engine.voices().map_or(0, |v| v.iter().filter(|v| v.blend.is_some()).count());
                say(true, format!("{named} named voice(s) from {}", path.display()));
            }
            let missing = engine.missing();
            let megabytes = missing.iter().map(|f| f.bytes).sum::<u64>().div_ceil(1_000_000);
            match (missing.len(), args.engine.offline) {
                (0, _) => say(true, "every model file is cached".into()),
                (n, true) => {
                    say(false, format!("{n} model file(s) ({megabytes} MB) are not cached and --offline is set: run `loqui fetch`"))
                }
                (n, false) => say(true, format!("{n} model file(s) ({megabytes} MB) download on first use, or now with `loqui fetch`")),
            }
        }
        Err(e) => say(false, format!("serve would refuse to start: {e}")),
    }
    if problems == 0 { Ok(()) } else { Err(format!("{problems} warning(s)")) }
}

fn token(action: TokenAction) -> Result<(), String> {
    let path = fs::default_token_path().ok_or("no config directory: set HOME")?;
    match action {
        TokenAction::Path => println!("{}", path.display()),
        TokenAction::Show => {
            if !path.exists() {
                auth::generate_file(&path).map_err(|e| e.to_string())?;
            }
            auth::check_private_file(&path).map_err(|e| e.to_string())?;
            let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
            let first = text.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with('#')).ok_or("the token file is empty")?;
            println!("{first}");
        }
        TokenAction::Rotate => {
            auth::rotate(&path).map_err(|e| e.to_string())?;
            eprintln!("rotated {}; restart `loqui serve` and update clients", path.display());
        }
    }
    Ok(())
}

fn speak(args: SpeakArgs) -> Result<(), String> {
    let engine = build_engine(&args.engine, false, 4096, 0)?;
    let format = loqui::Format::parse(&args.format).map_err(|e| e.to_string())?;
    let request = loqui::SpeakRequest { text: args.text, voice: args.voice, speed: args.speed, format };
    let speech = engine.speak(&request).map_err(|e| e.to_string())?;
    std::fs::write(&args.output, &speech.audio).map_err(|e| format!("{}: {e}", args.output.display()))?;
    eprintln!("{:.1}s of speech written to {}", speech.duration_secs, args.output.display());
    Ok(())
}

#[cfg(feature = "whisper")]
fn transcribe(args: TranscribeArgs) -> Result<(), String> {
    let engine = build_engine(&args.engine, true, 4096, 7200)?;
    let audio = std::fs::read(&args.file).map_err(|e| format!("{}: {e}", args.file.display()))?;
    let request = loqui::TranscribeRequest { audio, language: args.language, ..Default::default() };
    let result = engine.transcribe(&request).map_err(|e| e.to_string())?;
    println!("{}", result.text);
    Ok(())
}

fn voices(file: VoicesFile) -> Result<(), String> {
    let mut builder = Engine::builder();
    for (name, spec) in file.read()? {
        builder = builder.voice(name, spec);
    }
    // Lazy: nothing is loaded or fetched to list voices.
    let voices = builder.build().and_then(|engine| engine.voices()).map_err(|e| e.to_string())?;
    let width = voices.iter().map(|v| v.id.len()).max().unwrap_or(0);
    for v in voices {
        let line = format!("{:width$}  {}  {:7}  {}", v.id, v.language, v.gender, v.blend.unwrap_or_default());
        println!("{}", line.trim_end());
    }
    Ok(())
}

fn fetch(args: EngineArgs) -> Result<(), String> {
    // Fetching verifies every digest but loads nothing, so preparing a cache
    // never holds a model in memory.
    let engine = build_engine(&EngineArgs { preload: false, ..args }, true, 4096, 0)?;
    engine.fetch().map_err(|e| e.to_string())?;
    for m in engine.models() {
        eprintln!("{} ready", m.id);
    }
    engine.fetch_voices().map_err(|e| e.to_string())?;
    eprintln!("{} voices ready", loqui_kokoro::ENGLISH_VOICES.len());
    Ok(())
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        // whisper.cpp narrates every model load at info; keep it to warnings
        // unless RUST_LOG asks for more.
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,whisper_rs=warn".into()))
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Serve(args) => match tokio::runtime::Runtime::new() {
            Ok(rt) => rt.block_on(serve(args)),
            Err(e) => Err(e.to_string()),
        },
        Command::Speak(args) => speak(args),
        #[cfg(feature = "whisper")]
        Command::Transcribe(args) => transcribe(args),
        Command::Token { action } => token(action),
        Command::Doctor(args) => doctor(&args),
        Command::Fetch(args) => fetch(args),
        Command::Voices(file) => voices(file),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("loqui: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("loqui-cli-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn file(&self, name: &str, text: &str) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, text).unwrap();
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn voices_files_are_name_spec_pairs() {
        let dir = TempDir::new("voices-ok");
        let path = dir.file("voices.toml", "# my voices\nwill = \"am_puck(1)+am_liam(1)+am_onyx(0.5)\"\nnova = \"nova(3)+af_sky\"\n");
        let mut voices = read_voices(&path).unwrap();
        voices.sort();
        assert_eq!(voices, [("nova".into(), "nova(3)+af_sky".into()), ("will".into(), "am_puck(1)+am_liam(1)+am_onyx(0.5)".into())]);
    }

    #[test]
    fn voices_files_say_what_is_wrong_and_where() {
        let dir = TempDir::new("voices-bad");
        let table = read_voices(&dir.file("table.toml", "[will]\nblend = \"am_puck\"\n")).unwrap_err();
        assert!(table.contains("table.toml") && table.contains("\"will\" must be a string"), "{table}");
        let syntax = read_voices(&dir.file("syntax.toml", "will = am_puck\n")).unwrap_err();
        assert!(syntax.contains("syntax.toml") && syntax.contains("line 1"), "{syntax}");
        let missing = read_voices(&dir.0.join("absent.toml")).unwrap_err();
        assert!(missing.contains("absent.toml"), "{missing}");
    }

    #[test]
    fn the_default_voices_file_is_used_only_if_it_exists() {
        let dir = TempDir::new("voices-default");
        let explicit = dir.0.join("mine.toml");
        assert_eq!(voices_path(Some(&explicit), Some(&dir.0)), Some(explicit.clone()), "named files are used even if absent");
        assert_eq!(voices_path(None, Some(&dir.0)), None);
        assert_eq!(voices_path(None, None), None);
        let default = dir.file("voices.toml", "");
        assert_eq!(voices_path(None, Some(&dir.0)), Some(default));
    }
}
