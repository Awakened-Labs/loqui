//! Whisper speech-to-text on whisper.cpp.
//!
//! Decoding follows what open-speech asks of faster-whisper: beam search of
//! width 5, temperature 0, an optional initial prompt, and a language that
//! is detected unless given. Input is 16 kHz mono `f32`
//! (`loqui_audio::decode_mono_16k` produces it from any upload).
//!
//! Weights are GGML files such as `ggml-large-v3-turbo.bin` from
//! `ggerganov/whisper.cpp`; they are not bundled.

use std::path::Path;
use std::sync::Mutex;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

/// whisper.cpp timestamps are in hundredths of a second.
const TIMESTAMP_UNIT_SECS: f64 = 0.01;
const SAMPLE_RATE: f64 = 16_000.0;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("loading Whisper model: {0}")]
    Load(String),
    #[error("transcription failed: {0}")]
    Inference(String),
    #[error("language {0:?} is not supported by this English-only model")]
    Language(String),
}

/// Where inference runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    /// A GPU by ordinal. Needs the `cuda` or `vulkan` feature; without one,
    /// loading fails rather than silently running on the CPU.
    Gpu(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Task {
    #[default]
    Transcribe,
    /// Transcribe into English, whatever the spoken language.
    Translate,
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    /// An ISO 639-1 code such as `"en"`; `None` or `"auto"` detects it.
    pub language: Option<String>,
    /// Text that primes vocabulary and style (Whisper's initial prompt).
    pub prompt: Option<String>,
    pub task: Task,
    /// 0 decodes deterministically, which is open-speech's default.
    pub temperature: f32,
    /// CPU threads; `None` lets whisper.cpp choose.
    pub threads: Option<u16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Seconds from the start of the audio.
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transcription {
    /// Segment texts joined, as faster-whisper joins them.
    pub text: String,
    /// The language spoken (requested or detected), when known.
    pub language: Option<String>,
    /// Length of the audio in seconds.
    pub duration: f64,
    pub segments: Vec<Segment>,
}

pub struct Whisper {
    // The context owns the weights; the state is the scratch space for one
    // inference at a time. Field order drops the state first.
    state: Mutex<WhisperState>,
    _context: WhisperContext,
    multilingual: bool,
}

impl std::fmt::Debug for Whisper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Whisper").field("multilingual", &self.multilingual).finish_non_exhaustive()
    }
}

impl Whisper {
    /// Loads a GGML model file.
    pub fn load(path: &Path, device: Device) -> Result<Self, Error> {
        whisper_rs::install_logging_hooks();
        let mut params = WhisperContextParameters::default();
        match device {
            Device::Cpu => {
                params.use_gpu(false);
            }
            Device::Gpu(ordinal) => {
                if !cfg!(any(feature = "cuda", feature = "vulkan")) {
                    return Err(Error::Load(
                        "a GPU was requested but loqui-whisper was built without the `cuda` or `vulkan` feature".into(),
                    ));
                }
                params.use_gpu(true).gpu_device(ordinal);
            }
        }
        let path_str = path.to_str().ok_or_else(|| Error::Load(format!("{} is not UTF-8", path.display())))?;
        let context = WhisperContext::new_with_params(path_str, params)
            .map_err(|e| Error::Load(format!("{}: {e}", path.display())))?;
        let state = context.create_state().map_err(|e| Error::Load(e.to_string()))?;
        let multilingual = context.is_multilingual();
        Ok(Self { state: Mutex::new(state), _context: context, multilingual })
    }

    pub fn is_multilingual(&self) -> bool {
        self.multilingual
    }

    /// Transcribes 16 kHz mono audio. One call at a time per model; others
    /// wait.
    pub fn transcribe(&self, samples: &[f32], options: &Options) -> Result<Transcription, Error> {
        let language = options.language.as_deref().filter(|l| !l.eq_ignore_ascii_case("auto"));
        if let Some(lang) = language
            && !self.multilingual
            && !lang.eq_ignore_ascii_case("en")
        {
            return Err(Error::Language(lang.to_owned()));
        }

        let mut params = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 5, patience: -1.0 });
        params.set_language(language);
        params.set_translate(options.task == Task::Translate);
        params.set_temperature(options.temperature);
        if let Some(prompt) = options.prompt.as_deref().filter(|p| !p.is_empty()) {
            params.set_initial_prompt(prompt);
        }
        if let Some(threads) = options.threads.filter(|t| *t > 0) {
            params.set_n_threads(i32::from(threads));
        }
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        let mut state = self.state.lock().map_err(|_| Error::Inference("model lock poisoned".into()))?;
        state.full(params, samples).map_err(|e| Error::Inference(e.to_string()))?;

        let mut segments = Vec::new();
        let mut text = String::new();
        for segment in state.as_iter() {
            let piece = segment.to_str_lossy().map_err(|e| Error::Inference(e.to_string()))?;
            text.push_str(&piece);
            let start = segment.start_timestamp() as f64 * TIMESTAMP_UNIT_SECS;
            let end = (segment.end_timestamp() as f64 * TIMESTAMP_UNIT_SECS).max(start);
            if !piece.trim().is_empty() {
                segments.push(Segment { start, end, text: piece.into_owned() });
            }
        }
        let detected = whisper_rs::get_lang_str(state.full_lang_id_from_state()).map(str::to_owned);
        let language = match language {
            Some(lang) => Some(lang.to_owned()),
            None if !self.multilingual => Some("en".to_owned()),
            None => detected,
        };
        Ok(Transcription {
            text: text.trim().to_owned(),
            language,
            duration: samples.len() as f64 / SAMPLE_RATE,
            segments,
        })
    }
}
