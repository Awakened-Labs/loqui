//! Kokoro-82M text-to-speech, in process.
//!
//! Text goes through [`loqui_g2p`] (misaki's English G2P, GPL-free), is cut
//! into chunks of at most 510 phonemes exactly as Kokoro's own `KPipeline`
//! cuts it, and each chunk runs through the ONNX export of Kokoro on ONNX
//! Runtime. The result is 24 kHz mono `f32`, trimmed of edge silence and
//! peak-normalised the way open-speech delivers it.
//!
//! Weights are not bundled. Point [`KokoroModel::load`] at
//! `onnx/model.onnx` and [`Kokoro::new`] at the `voices/` directory of
//! `onnx-community/Kokoro-82M-v1.0-ONNX` (Apache-2.0).

mod blend;
mod chunk;
mod model;
pub mod post;
mod voice;

use std::path::PathBuf;
use std::sync::OnceLock;

use loqui_g2p::{Dialect, G2p, OovFallback};

pub use blend::Blend;
pub use chunk::{chunks, tokens_to_ps};
pub use model::{Device, KokoroModel, SAMPLE_RATE};
use voice::VoicePacks;
pub use voice::{MAX_PHONEMES, STYLE_DIM, Voice};

/// Kokoro's default voice, and what OpenAI's `alloy` maps to.
pub const DEFAULT_VOICE: &str = "af_heart";

/// Kokoro v1.0's American (`a*`) and British (`b*`) voices: the ones this
/// crate can phonemize for. `f`/`m` in the second letter is the speaker's
/// voice type as Kokoro labels it.
pub const ENGLISH_VOICES: &[&str] = &[
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
];

/// Non-exhaustive so that a new failure need not be a breaking change.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("model: {0}")]
    Model(String),
    #[error("voice: {0}")]
    Voice(String),
    #[error("G2P: {0}")]
    G2p(#[from] loqui_g2p::Error),
    #[error("speed {0} is outside 0.25..=4.0")]
    Speed(f32),
}

/// OpenAI's voice names, mapped the way open-speech maps them, so a client
/// written against OpenAI keeps working. [`Blend`] applies this to each
/// part of a spec.
pub fn resolve_alias(voice: &str) -> &str {
    match voice {
        "alloy" => "af_heart",
        "echo" => "am_adam",
        "fable" => "bf_emma",
        "onyx" => "am_michael",
        "nova" => "af_nova",
        "shimmer" => "af_bella",
        other => other,
    }
}

/// Text in, audio out. `Send + Sync`: share one per process.
pub struct Kokoro {
    model: KokoroModel,
    fallback: OovFallback,
    american: OnceLock<G2p>,
    british: OnceLock<G2p>,
    packs: VoicePacks,
}

impl std::fmt::Debug for Kokoro {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kokoro").field("model", &self.model).field("voices_dir", &self.packs.dir()).finish_non_exhaustive()
    }
}

impl Kokoro {
    pub fn new(model: KokoroModel, voices_dir: PathBuf, fallback: OovFallback) -> Self {
        Self { model, fallback, american: OnceLock::new(), british: OnceLock::new(), packs: VoicePacks::new(voices_dir) }
    }

    /// Speaks `text` in `voice` at `speed` (0.25 to 4.0). Returns 24 kHz
    /// mono samples, edge silence trimmed and peak-normalised to 0.95, as
    /// open-speech returns them. A voice is any pack in the voices
    /// directory, or a blend of them: `&"af_bella(2)+af_sky(1)".parse()?`.
    pub fn speak(&self, text: &str, voice: &Blend, speed: f32) -> Result<Vec<f32>, Error> {
        let audio = self.speak_raw(text, voice, speed)?;
        Ok(post::normalize_peak(post::trim_silence(&audio, post::TRIM_THRESHOLD), post::PEAK))
    }

    /// [`Kokoro::speak`] without trimming or normalisation: the model's
    /// output, chunks concatenated.
    pub fn speak_raw(&self, text: &str, voice: &Blend, speed: f32) -> Result<Vec<f32>, Error> {
        if !(0.25..=4.0).contains(&speed) {
            return Err(Error::Speed(speed));
        }
        let dialect = Dialect::for_kokoro_voice(voice.lead())
            .ok_or_else(|| Error::Voice("only American (a*) and British (b*) voices are supported".into()))?;
        let style = self.packs.get(voice)?;
        let mut audio = Vec::new();
        for ps in self.phoneme_chunks(text, dialect)? {
            audio.extend(self.model.synthesize(&ps, &style, speed)?);
        }
        Ok(audio)
    }

    /// The phoneme chunks `text` is spoken as. `KPipeline` splits on
    /// newlines before phonemizing, then chunks each line.
    pub fn phoneme_chunks(&self, text: &str, dialect: Dialect) -> Result<Vec<String>, Error> {
        let g2p = self.g2p(dialect)?;
        Ok(text.trim().split('\n').filter(|line| !line.trim().is_empty()).flat_map(|line| chunks(g2p.tokens(line))).collect())
    }

    fn g2p(&self, dialect: Dialect) -> Result<&G2p, Error> {
        let cell = match dialect {
            Dialect::American => &self.american,
            Dialect::British => &self.british,
        };
        if let Some(g2p) = cell.get() {
            return Ok(g2p);
        }
        let built = G2p::new(dialect, self.fallback)?;
        Ok(cell.get_or_init(|| built))
    }
}
