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
//!
//! Before shipping an embedded engine, read
//! [Embedding loqui safely](https://github.com/Awakened-Labs/loqui/blob/main/docs/embedding.md): downloads, cache permissions, limits on
//! untrusted input, and the licensing each feature brings (leave `mp3` off in
//! proprietary programs).

mod models;
mod slot;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::time::Duration;

pub use loqui_audio::Format;
pub use loqui_g2p::OovFallback;
pub use loqui_kokoro::Blend;
pub use models::{Downloads, KokoroVariant, WHISPER_MODELS, default_cache_dir};

use loqui_kokoro::{Kokoro, KokoroModel};
use slot::Slot;

/// The id `/v1/models` lists for the Kokoro model.
pub const TTS_MODEL_ID: &str = "kokoro";

/// Non-exhaustive because `Stt` exists only with the `whisper` feature.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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
            || matches!(
                self,
                Self::Audio(
                    loqui_audio::Error::Decode(_) | loqui_audio::Error::TooLong { .. } | loqui_audio::Error::UnsupportedFormat { .. }
                )
            )
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
    voices: Vec<(String, String)>,
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

    /// Names a voice: afterwards `name` may be spoken in, alone or inside a
    /// blend, as if it were a built-in voice. `spec` is a blend of built-in
    /// voices, such as `am_puck(1)+am_liam(1)+am_onyx(0.5)`.
    ///
    /// Names are lowercase ASCII letters, digits and `_`. A name may take
    /// over an OpenAI name (`nova`), so clients that offer only OpenAI's
    /// voices can reach it, but not a Kokoro-shaped one (`af_custom`), which
    /// stays reserved for Kokoro's own voices. [`EngineBuilder::build`]
    /// checks every name and spec.
    pub fn voice(mut self, name: impl Into<String>, spec: impl Into<String>) -> Self {
        self.voices.push((name.into(), spec.into()));
        self
    }

    pub fn build(self) -> Result<Engine, Error> {
        let names = named_voices(self.voices)?;
        let cache = match self.cache_dir {
            Some(dir) => dir,
            None => {
                default_cache_dir().ok_or_else(|| Error::Config("no cache directory: set HOME or XDG_CACHE_HOME, or pass one".into()))?
            }
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
            names,
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
    names: BTreeMap<String, Named>,
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

/// A voice an application has named: its spec as written, and what that
/// resolves to.
struct Named {
    spec: String,
    blend: Blend,
}

/// Checks named voices. They are made only of built-in voices, never of
/// each other, so there are no cycles and their order does not matter.
/// Errors name the voice: it is the operator's configuration, not a
/// caller's input.
fn named_voices(voices: Vec<(String, String)>) -> Result<BTreeMap<String, Named>, Error> {
    let defined: Vec<String> = voices.iter().map(|(name, _)| name.clone()).collect();
    let mut names = BTreeMap::new();
    for (name, spec) in voices {
        let bad = |why: &str| Error::Config(format!("voice {name:?}: {why}"));
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_') {
            return Err(bad("names may contain only lowercase ASCII letters, digits and _"));
        }
        if is_kokoro_shaped(&name) {
            return Err(bad("names shaped like Kokoro's (af_, bm_, ...) are reserved for its own voices"));
        }
        let blend = Blend::parse(&spec, |_| None).map_err(|e| bad(&e.to_string()))?;
        if let Some((id, _)) = blend.parts().iter().find(|(id, _)| !loqui_kokoro::ENGLISH_VOICES.contains(&id.as_str())) {
            return Err(bad(&if defined.contains(id) {
                format!("{id} is a named voice; named voices are made of built-in voices only")
            } else {
                format!("{id} is not a built-in voice")
            }));
        }
        if names.insert(name.clone(), Named { spec: spec.trim().to_owned(), blend }).is_some() {
            return Err(bad("named twice"));
        }
    }
    Ok(names)
}

/// A letter, `f` or `m`, then `_`: how every Kokoro voice is named.
fn is_kokoro_shaped(name: &str) -> bool {
    matches!(name.as_bytes(), [lang, b'f' | b'm', b'_', ..] if lang.is_ascii_lowercase())
}

/// A voice [`Engine::speak`] accepts, as open-speech's `/v1/audio/voices`
/// describes one.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct VoiceInfo {
    /// What to pass as [`SpeakRequest::voice`].
    pub id: String,
    /// `Heart` for `af_heart`; a named voice's own name.
    pub name: String,
    /// The accent it speaks with: `en-us` or `en-gb`.
    pub language: &'static str,
    /// `female` or `male`, or `unknown` for a blend of both.
    pub gender: &'static str,
    /// A named voice's spec, as configured; `None` for a built-in voice.
    pub blend: Option<String>,
}

impl VoiceInfo {
    fn builtin(id: &str) -> Self {
        let mut name = id.get(3..).unwrap_or(id).to_owned();
        if let Some(initial) = name.get_mut(..1) {
            initial.make_ascii_uppercase();
        }
        Self { id: id.to_owned(), name, language: language(id), gender: gender(id), blend: None }
    }

    fn named(name: &str, voice: &Named) -> Self {
        let mut genders = voice.blend.audible().map(|(id, _)| gender(id));
        let first = genders.next().unwrap_or("unknown");
        let gender = if genders.all(|g| g == first) { first } else { "unknown" };
        Self { id: name.to_owned(), name: name.to_owned(), language: language(voice.blend.lead()), gender, blend: Some(voice.spec.clone()) }
    }
}

fn language(id: &str) -> &'static str {
    if id.starts_with('b') { "en-gb" } else { "en-us" }
}

fn gender(id: &str) -> &'static str {
    match id.as_bytes().get(1) {
        Some(b'f') => "female",
        Some(b'm') => "male",
        _ => "unknown",
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
    /// A voice spec: `af_heart`, an OpenAI name (`alloy`), or a blend
    /// (`af_bella(2)+af_sky(1)`; see [`Blend`]).
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
pub use loqui_whisper::{Device as SttDevice, Segment, Task, Transcription};

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
            voices: Vec::new(),
        }
    }

    /// Loads every enabled model now, and fetches the voices that named
    /// voices are made of, so a host running with [`Downloads::Deny`] finds
    /// a missing one at startup rather than on the first request.
    pub fn preload(&self) -> Result<(), Error> {
        if let Some(slot) = &self.inner.tts {
            slot.get()?;
            for named in self.inner.names.values() {
                named.blend.audible().try_for_each(|(id, _)| self.fetch_voice(id))?;
            }
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

    /// The voices [`Engine::speak`] accepts by name: the built-in English
    /// voices, then any named with [`EngineBuilder::voice`]. Any blend of
    /// these is accepted too.
    pub fn voices(&self) -> Result<Vec<VoiceInfo>, Error> {
        self.inner.tts.as_ref().ok_or(Error::Disabled("text-to-speech"))?;
        let builtin = loqui_kokoro::ENGLISH_VOICES.iter().map(|id| VoiceInfo::builtin(id));
        Ok(builtin.chain(self.inner.names.iter().map(|(name, voice)| VoiceInfo::named(name, voice))).collect())
    }

    /// Downloads every voice pack on the roster now (about 15 MB), so any
    /// voice or blend can be spoken on a host that later runs with
    /// [`Downloads::Deny`].
    pub fn fetch_voices(&self) -> Result<(), Error> {
        self.inner.tts.as_ref().ok_or(Error::Disabled("text-to-speech"))?;
        loqui_kokoro::ENGLISH_VOICES.iter().try_for_each(|id| self.fetch_voice(id))
    }

    /// A voice spec as this engine reads it: named voices first, then
    /// OpenAI's names, then pack ids.
    fn resolve(&self, spec: &str) -> Result<Blend, Error> {
        Ok(Blend::parse(spec, |id| self.inner.names.get(id).map(|named| &named.blend))?)
    }

    fn fetch_voice(&self, id: &str) -> Result<(), Error> {
        models::fetch(&self.inner.cache, &models::KOKORO_REPO, &voice_file(id), None, self.inner.downloads).map(drop)
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
        // Every part is checked before any is fetched, and only voices on
        // the roster are fetched, so a request cannot make the engine
        // download arbitrary files, nor anything at all for a bad voice.
        let voice = self.resolve(&request.voice)?;
        if voice.parts().iter().any(|(id, _)| !loqui_kokoro::ENGLISH_VOICES.contains(&id.as_str())) {
            // The id is not echoed: it is caller input that failed validation.
            return Err(Error::Invalid("unknown voice".into()));
        }
        voice.audible().try_for_each(|(id, _)| self.fetch_voice(id))?;
        let kokoro = slot.get()?;
        let samples = kokoro.speak(&request.text, &voice, request.speed)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// An engine that loads nothing: building is lazy until something
    /// is spoken.
    fn engine(voices: &[(&str, &str)]) -> Result<Engine, Error> {
        let builder = Engine::builder().cache_dir(std::env::temp_dir().join("loqui-unused-cache")).downloads(Downloads::Deny);
        voices.iter().fold(builder, |b, (name, spec)| b.voice(*name, *spec)).build()
    }

    fn close(blend: &Blend, expected: &[(&str, f32)]) -> bool {
        let parts = blend.parts();
        parts.len() == expected.len() && parts.iter().zip(expected).all(|((a, x), (b, y))| a == b && (x - y).abs() < 1e-6)
    }

    #[test]
    fn named_voices_are_checked_when_the_engine_is_built() {
        let refused = [
            ("Will", "am_puck"),
            ("will smith", "am_puck"),
            ("", "am_puck"),
            ("af_custom", "am_puck"),
            ("jf_alpha", "am_puck"),
            ("will", "am_puck(x)"),
            ("will", "zz_nobody"),
            ("will", "jf_alpha"),
        ];
        for (name, spec) in refused {
            let err = engine(&[(name, spec)]).expect_err(&format!("{name:?} = {spec:?}"));
            assert!(matches!(err, Error::Config(_)), "{name:?} = {spec:?}: {err}");
        }
        assert!(matches!(engine(&[("will", "am_puck"), ("will", "am_liam")]), Err(Error::Config(_))));
        let err = engine(&[("will", "am_puck"), ("bill", "will(2)+am_liam")]).unwrap_err().to_string();
        assert!(err.contains("built-in voices only"), "{err}");
    }

    #[test]
    fn named_voices_resolve_alone_and_inside_blends() {
        let engine = engine(&[("will", "am_puck(1)+am_liam(1)+am_onyx(0.5)")]).unwrap();
        assert!(close(&engine.resolve("will").unwrap(), &[("am_puck", 0.4), ("am_liam", 0.4), ("am_onyx", 0.2)]));
        let blend = engine.resolve("will(2)+af_sky(1)").unwrap();
        assert!(close(&blend, &[("am_puck", 0.8 / 3.0), ("am_liam", 0.8 / 3.0), ("am_onyx", 0.4 / 3.0), ("af_sky", 1.0 / 3.0)]));
    }

    #[test]
    fn a_name_can_take_over_an_openai_name_and_use_it() {
        let engine = engine(&[("nova", "nova(3)+af_sky(1)")]).unwrap();
        assert!(close(&engine.resolve("nova").unwrap(), &[("af_nova", 0.75), ("af_sky", 0.25)]));
        assert!(close(&engine.resolve("alloy").unwrap(), &[("af_heart", 1.0)]));
    }

    #[test]
    fn unknown_voices_are_refused_without_echoing_them() {
        let engine = engine(&[]).unwrap();
        let err = engine.speak(&SpeakRequest { voice: "af_heart+zz_secret".into(), ..SpeakRequest::new("hello") }).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)), "{err}");
        assert!(!err.to_string().contains("secret"), "{err}");
    }

    #[test]
    fn voices_lists_the_roster_then_named_voices() {
        let engine = engine(&[("will", "am_puck(1)+am_liam(1)"), ("duo", " af_sky + am_adam "), ("lady", "bf_emma(0)+af_sky")]).unwrap();
        let voices = engine.voices().unwrap();
        assert_eq!(voices.len(), loqui_kokoro::ENGLISH_VOICES.len() + 3);
        let heart = voices.iter().find(|v| v.id == "af_heart").unwrap();
        assert_eq!((heart.name.as_str(), heart.language, heart.gender, heart.blend.as_deref()), ("Heart", "en-us", "female", None));
        let george = voices.iter().find(|v| v.id == "bm_george").unwrap();
        assert_eq!((george.name.as_str(), george.language, george.gender), ("George", "en-gb", "male"));
        let named: Vec<_> = voices[loqui_kokoro::ENGLISH_VOICES.len()..].iter().map(|v| (v.id.as_str(), v.language, v.gender)).collect();
        // Sorted by name; a weightless lead sets the accent but not the gender.
        assert_eq!(named, [("duo", "en-us", "unknown"), ("lady", "en-gb", "female"), ("will", "en-us", "male")]);
        assert_eq!(voices.last().unwrap().blend.as_deref(), Some("am_puck(1)+am_liam(1)"));
    }

    #[test]
    fn voices_needs_text_to_speech() {
        let engine = Engine::builder().cache_dir(std::env::temp_dir()).tts(None).build().unwrap();
        assert!(matches!(engine.voices(), Err(Error::Disabled(_))));
    }
}
