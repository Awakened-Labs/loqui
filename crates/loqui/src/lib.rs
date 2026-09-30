//! Local speech for Rust: Kokoro text-to-speech and Whisper speech-to-text.
//!
//! An [`Engine`] runs both models in process. It opens no sockets and
//! listens on nothing; the only network access is fetching model weights,
//! pinned to commit revisions, and that can be switched off with
//! [`Downloads::Deny`]. To serve speech to other processes, see the
//! `loqui-server` crate, which puts an authenticated, explicitly exposed
//! HTTP API in front of an `Engine`.
//!
//! ```no_run
//! use loqui::{Engine, SpeakRequest};
//!
//! let engine = Engine::builder().build()?;
//! let speech = engine.speak(&SpeakRequest::new("Hello from loqui."))?;
//! std::fs::write("hello.wav", &speech.audio)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Models load on first use and, if given an idle timeout, unload when
//! unused. Everything is synchronous; call it from `spawn_blocking` in async
//! code.

mod models;
mod slot;

use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::time::Duration;

pub use loqui_audio::Format;
pub use loqui_g2p::OovFallback;
pub use models::{Downloads, KokoroVariant, WHISPER_MODELS, default_cache_dir};

use loqui_kokoro::{Kokoro, KokoroModel};
use slot::Slot;

/// The id `/v1/models` lists for the Kokoro model.
pub const TTS_MODEL_ID: &str = "kokoro";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The request itself is invalid (empty text, unknown voice, bad speed).
    #[error("{0}")]
    Invalid(String),
    #[error("input is {chars} characters; the limit is {limit}")]
    InputTooLong { chars: usize, limit: usize },
    /// The engine was built without this capability.
    #[error("{0} is not enabled in this engine")]
    Disabled(&'static str),
    #[error("configuration: {0}")]
    Config(String),
    #[error("model unavailable: {0}")]
    Unavailable(String),
    #[error("weights missing: {0}")]
    Missing(String),
    #[error("download failed: {0}")]
    Download(String),
    #[error("integrity check failed: {0}")]
    Integrity(String),
    #[error("{0}")]
    Io(String),
    #[error("audio: {0}")]
    Audio(#[from] loqui_audio::Error),
    #[error("speech synthesis: {0}")]
    Tts(#[from] loqui_kokoro::Error),
    #[cfg(feature = "whisper")]
    #[error("transcription: {0}")]
    Stt(#[from] loqui_whisper::Error),
}

impl Error {
    /// Whether the caller's input caused this, as opposed to the engine.
    /// Servers map this to 4xx rather than 5xx.
    pub fn is_client_error(&self) -> bool {
        matches!(self, Self::Invalid(_) | Self::InputTooLong { .. } | Self::Disabled(_))
            || matches!(self, Self::Audio(loqui_audio::Error::Decode(_) | loqui_audio::Error::TooLong { .. } | loqui_audio::Error::UnsupportedFormat(_)))
            || matches!(self, Self::Tts(loqui_kokoro::Error::Voice(_) | loqui_kokoro::Error::Speed(_)))
    }
}

pub use loqui_kokoro::Device as TtsDevice;

/// Text-to-speech settings.
#[derive(Debug, Clone)]
pub struct TtsConfig {
    pub variant: KokoroVariant,
    pub device: TtsDevice,
    /// CPU threads for ONNX Runtime; `None` lets it choose.
    pub threads: Option<usize>,
    pub fallback: OovFallback,
    /// Unload after this long unused; `None` keeps the model resident.
    /// Kokoro is small (about 330 MB on CPU), so resident is the default.
    pub idle_ttl: Option<Duration>,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self { variant: KokoroVariant::Fp32, device: TtsDevice::Cpu, threads: None, fallback: OovFallback::Neural, idle_ttl: None }
    }
}

/// Speech-to-text settings.
#[cfg(feature = "whisper")]
#[derive(Debug, Clone)]
pub struct SttConfig {
    /// A name from [`WHISPER_MODELS`], e.g. `"large-v3-turbo"`.
    pub model: String,
    pub device: loqui_whisper::Device,
    /// Unload after this long unused; `None` keeps it resident. Whisper
    /// large is 1.6 GB, so the default frees it after ten idle minutes.
    pub idle_ttl: Option<Duration>,
}

#[cfg(feature = "whisper")]
impl Default for SttConfig {
    fn default() -> Self {
        Self { model: "large-v3-turbo".into(), device: loqui_whisper::Device::Cpu, idle_ttl: Some(Duration::from_secs(600)) }
    }
}

pub struct EngineBuilder {
    cache_dir: Option<PathBuf>,
    downloads: Downloads,
    tts: Option<TtsConfig>,
    #[cfg(feature = "whisper")]
    stt: Option<SttConfig>,
    max_input_chars: usize,
    max_audio_secs: u64,
    preload: bool,
}

impl EngineBuilder {
    /// Where weights are cached. Defaults to [`default_cache_dir`].
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    pub fn downloads(mut self, downloads: Downloads) -> Self {
        self.downloads = downloads;
        self
    }

    /// Text-to-speech settings; `None` disables it.
    pub fn tts(mut self, config: Option<TtsConfig>) -> Self {
        self.tts = config;
        self
    }

    /// Speech-to-text settings; `None` (the default) disables it.
    #[cfg(feature = "whisper")]
    pub fn stt(mut self, config: Option<SttConfig>) -> Self {
        self.stt = config;
        self
    }

    /// The longest text [`Engine::speak`] accepts, in characters. 4096 by
    /// default, OpenAI's limit.
    pub fn max_input_chars(mut self, chars: usize) -> Self {
        self.max_input_chars = chars;
        self
    }

    /// The longest audio [`Engine::transcribe`] accepts. Enforced while
    /// decoding, so a small compressed upload cannot expand without bound.
    pub fn max_audio_secs(mut self, secs: u64) -> Self {
        self.max_audio_secs = secs;
        self
    }

    /// Load every enabled model during [`EngineBuilder::build`] instead of
    /// on first use.
    pub fn preload(mut self, preload: bool) -> Self {
        self.preload = preload;
        self
    }

    pub fn build(self) -> Result<Engine, Error> {
        let cache = match self.cache_dir {
            Some(dir) => dir,
            None => default_cache_dir().ok_or_else(|| Error::Config("no cache directory: set HOME or XDG_CACHE_HOME, or pass one".into()))?,
        };
        let downloads = self.downloads;

        let tts = self.tts.map(|cfg| {
            let cache = cache.clone();
            let ttl = cfg.idle_ttl;
            Slot::new(ttl, Box::new(move || load_kokoro(&cache, &cfg, downloads)))
        });
        #[cfg(feature = "whisper")]
        if let Some(cfg) = &self.stt {
            models::whisper_file(&cfg.model)?;
        }
        #[cfg(feature = "whisper")]
        let stt_model = self.stt.as_ref().map(|cfg| cfg.model.clone());
        #[cfg(not(feature = "whisper"))]
        let stt_model = None;
        #[cfg(feature = "whisper")]
        let stt = self.stt.map(|cfg| {
            let cache = cache.clone();
            let ttl = cfg.idle_ttl;
            Slot::new(ttl, Box::new(move || load_whisper(&cache, &cfg, downloads)))
        });

        let inner = Arc::new(Inner {
            cache,
            downloads,
            tts,
            #[cfg(feature = "whisper")]
            stt,
            stt_model,
            max_input_chars: self.max_input_chars,
            max_audio_secs: self.max_audio_secs,
        });
        if inner.has_ttl() {
            start_reaper(Arc::downgrade(&inner));
        }
        let engine = Engine { inner };
        if self.preload {
            engine.preload()?;
        }
        Ok(engine)
    }
}

struct Inner {
    cache: PathBuf,
    downloads: Downloads,
    tts: Option<Slot<Kokoro>>,
    #[cfg(feature = "whisper")]
    stt: Option<Slot<loqui_whisper::Whisper>>,
    stt_model: Option<String>,
    max_input_chars: usize,
    // Only transcription decodes uploads.
    #[cfg_attr(not(feature = "whisper"), allow(dead_code))]
    max_audio_secs: u64,
}

impl Inner {
    fn has_ttl(&self) -> bool {
        let tts = self.tts.as_ref().is_some_and(|s| s.ttl().is_some());
        #[cfg(feature = "whisper")]
        let stt = self.stt.as_ref().is_some_and(|s| s.ttl().is_some());
        #[cfg(not(feature = "whisper"))]
        let stt = false;
        tts || stt
    }

    fn reap(&self) {
        if let Some(slot) = &self.tts
            && slot.reap()
        {
            tracing::info!(kind = "tts", "idle model unloaded");
        }
        #[cfg(feature = "whisper")]
        if let Some(slot) = &self.stt
            && slot.reap()
        {
            tracing::info!(kind = "stt", "idle model unloaded");
        }
    }
}

/// A background thread that unloads idle models. It holds only a weak
/// reference and exits once the engine is dropped.
fn start_reaper(inner: Weak<Inner>) {
    let spawned = std::thread::Builder::new().name("loqui-reaper".into()).spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(15));
            match inner.upgrade() {
                Some(inner) => inner.reap(),
                None => return,
            }
        }
    });
    if let Err(e) = spawned {
        tracing::warn!("could not start the idle-unload thread, models will stay resident: {e}");
    }
}

fn load_kokoro(cache: &std::path::Path, cfg: &TtsConfig, downloads: Downloads) -> Result<Kokoro, Error> {
    let (file, sha) = cfg.variant.file();
    let onnx = models::fetch(cache, &models::KOKORO_REPO, file, Some(sha), downloads)?;
    // Fetching the default voice also tells us where voices live.
    let default_voice = models::fetch(cache, &models::KOKORO_REPO, &voice_file(loqui_kokoro::DEFAULT_VOICE), None, downloads)?;
    let voices_dir = default_voice.parent().map(PathBuf::from).ok_or_else(|| Error::Io("voice path has no parent".into()))?;
    let model = KokoroModel::load(&onnx, cfg.device, cfg.threads)?;
    tracing::info!(variant = ?cfg.variant, device = ?cfg.device, "Kokoro loaded");
    Ok(Kokoro::new(model, voices_dir, cfg.fallback))
}

#[cfg(feature = "whisper")]
fn load_whisper(cache: &std::path::Path, cfg: &SttConfig, downloads: Downloads) -> Result<loqui_whisper::Whisper, Error> {
    let (file, sha) = models::whisper_file(&cfg.model)?;
    let path = models::fetch(cache, &models::WHISPER_REPO, file, Some(sha), downloads)?;
    let whisper = loqui_whisper::Whisper::load(&path, cfg.device)?;
    tracing::info!(model = cfg.model, device = ?cfg.device, "Whisper loaded");
    Ok(whisper)
}

fn voice_file(id: &str) -> String {
    format!("voices/{id}.bin")
}

/// A text-to-speech request.
#[derive(Debug, Clone)]
pub struct SpeakRequest {
    pub text: String,
    /// A voice spec: `af_heart`, an OpenAI alias (`alloy`), or a blend
    /// (`af_bella(2)+af_sky(1)`).
    pub voice: String,
    /// 0.25 to 4.0.
    pub speed: f32,
    pub format: Format,
}

impl SpeakRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into(), voice: loqui_kokoro::DEFAULT_VOICE.into(), speed: 1.0, format: Format::Wav }
    }
}

/// Encoded speech.
#[derive(Debug, Clone)]
pub struct Speech {
    pub audio: Vec<u8>,
    pub format: Format,
    pub duration_secs: f64,
}

#[cfg(feature = "whisper")]
pub use loqui_whisper::{Segment, Task, Transcription};

/// A speech-to-text request.
#[cfg(feature = "whisper")]
#[derive(Debug, Clone, Default)]
pub struct TranscribeRequest {
    /// The uploaded file, in any container loqui-audio decodes.
    pub audio: Vec<u8>,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub task: Task,
    pub temperature: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Tts,
    Stt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    pub kind: ModelKind,
    pub loaded: bool,
}

/// Kokoro and Whisper, in process. Cheap to clone; clones share models.
#[derive(Clone)]
pub struct Engine {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine").field("cache", &self.inner.cache).field("models", &self.models()).finish()
    }
}

impl Engine {
    pub fn builder() -> EngineBuilder {
        EngineBuilder {
            cache_dir: None,
            downloads: Downloads::Allow,
            tts: Some(TtsConfig::default()),
            #[cfg(feature = "whisper")]
            stt: None,
            max_input_chars: 4096,
            max_audio_secs: 1800,
            preload: false,
        }
    }

    /// Loads every enabled model now.
    pub fn preload(&self) -> Result<(), Error> {
        if let Some(slot) = &self.inner.tts {
            slot.get()?;
        }
        #[cfg(feature = "whisper")]
        if let Some(slot) = &self.inner.stt {
            slot.get()?;
        }
        Ok(())
    }

    /// Unloads idle models now, whatever their TTL.
    pub fn unload(&self) {
        if let Some(slot) = &self.inner.tts {
            slot.unload();
        }
        #[cfg(feature = "whisper")]
        if let Some(slot) = &self.inner.stt {
            slot.unload();
        }
    }

    pub fn models(&self) -> Vec<ModelInfo> {
        let mut out = Vec::new();
        if let Some(slot) = &self.inner.tts {
            out.push(ModelInfo { id: TTS_MODEL_ID.into(), kind: ModelKind::Tts, loaded: slot.is_loaded() });
        }
        #[cfg(feature = "whisper")]
        if let (Some(slot), Some(id)) = (&self.inner.stt, &self.inner.stt_model) {
            out.push(ModelInfo { id: id.clone(), kind: ModelKind::Stt, loaded: slot.is_loaded() });
        }
        out
    }

    /// The configured Whisper model's name, if speech-to-text is enabled.
    pub fn stt_model_id(&self) -> Option<&str> {
        self.inner.stt_model.as_deref()
    }

    /// The English voices this engine can speak with.
    pub fn voices(&self) -> &'static [&'static str] {
        loqui_kokoro::ENGLISH_VOICES
    }

    pub fn speak(&self, request: &SpeakRequest) -> Result<Speech, Error> {
        let slot = self.inner.tts.as_ref().ok_or(Error::Disabled("text-to-speech"))?;
        let chars = request.text.chars().count();
        if request.text.trim().is_empty() {
            return Err(Error::Invalid("input text is empty".into()));
        }
        if chars > self.inner.max_input_chars {
            return Err(Error::InputTooLong { chars, limit: self.inner.max_input_chars });
        }
        if !(0.25..=4.0).contains(&request.speed) {
            return Err(Error::Invalid(format!("speed {} is outside 0.25 to 4.0", request.speed)));
        }
        // Only voices on the roster are fetched, so a request cannot make
        // the engine download arbitrary files.
        for (id, _) in loqui_kokoro::parse_voice_spec(&request.voice)? {
            if !loqui_kokoro::ENGLISH_VOICES.contains(&id.as_str()) {
                return Err(Error::Invalid(format!("unknown voice {id:?}")));
            }
            models::fetch(&self.inner.cache, &models::KOKORO_REPO, &voice_file(&id), None, self.inner.downloads)?;
        }
        let kokoro = slot.get()?;
        let samples = kokoro.speak(&request.text, &request.voice, request.speed)?;
        let pcm = loqui_audio::Pcm { samples, rate: loqui_kokoro::SAMPLE_RATE };
        let duration_secs = pcm.duration_secs();
        let audio = loqui_audio::encode(&pcm, request.format)?;
        Ok(Speech { audio, format: request.format, duration_secs })
    }

    #[cfg(feature = "whisper")]
    pub fn transcribe(&self, request: &TranscribeRequest) -> Result<Transcription, Error> {
        let slot = self.inner.stt.as_ref().ok_or(Error::Disabled("speech-to-text"))?;
        let pcm = loqui_audio::decode_mono_16k(&request.audio, Some(self.inner.max_audio_secs))?;
        let whisper = slot.get()?;
        let options = loqui_whisper::Options {
            language: request.language.clone(),
            prompt: request.prompt.clone(),
            task: request.task,
            temperature: request.temperature,
            threads: None,
        };
        Ok(whisper.transcribe(&pcm.samples, &options)?)
    }
}
