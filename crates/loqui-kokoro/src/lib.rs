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

mod chunk;
mod model;
pub mod post;
mod voice;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use loqui_g2p::{Dialect, G2p, OovFallback};

pub use chunk::{chunks, tokens_to_ps};
pub use model::{Device, KokoroModel, SAMPLE_RATE};
pub use voice::{MAX_PHONEMES, STYLE_DIM, Voice};

/// Kokoro's default voice, and what OpenAI's `alloy` maps to.
pub const DEFAULT_VOICE: &str = "af_heart";

/// Kokoro v1.0's American (`a*`) and British (`b*`) voices: the ones this
/// crate can phonemize for. `f`/`m` in the second letter is the speaker's
/// voice type as Kokoro labels it.
pub const ENGLISH_VOICES: &[&str] = &[
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river",
    "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
    "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george",
    "bm_lewis",
];

#[derive(Debug, thiserror::Error)]
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
/// written against OpenAI keeps working.
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

/// Parses a voice spec: `af_heart`, an OpenAI alias such as `alloy`, or a
/// blend such as `af_bella+af_sky` or `af_bella(2)+af_sky(1)`. Voice ids are
/// restricted to ASCII letters, digits and `_`, because they become file
/// names.
pub fn parse_voice_spec(spec: &str) -> Result<Vec<(String, f32)>, Error> {
    let spec = if spec.contains(['+', '(']) { spec } else { resolve_alias(spec) };
    spec.split('+')
        .map(|part| {
            let part = part.trim();
            let (id, weight) = match part.split_once('(') {
                Some((id, rest)) => {
                    let w = rest
                        .strip_suffix(')')
                        .and_then(|w| w.parse::<f32>().ok())
                        .filter(|w| w.is_finite() && *w >= 0.0);
                    (id, w.ok_or_else(|| Error::Voice(format!("bad weight in {part:?}")))?)
                }
                None => (part, 1.0),
            };
            if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(Error::Voice(format!("invalid voice id {id:?}")));
            }
            Ok((id.to_owned(), weight))
        })
        .collect()
}

/// Text in, audio out. `Send + Sync`: share one per process.
pub struct Kokoro {
    model: KokoroModel,
    voices_dir: PathBuf,
    fallback: OovFallback,
    american: OnceLock<G2p>,
    british: OnceLock<G2p>,
    voices: Mutex<HashMap<String, Arc<Voice>>>,
}

impl std::fmt::Debug for Kokoro {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kokoro").field("model", &self.model).field("voices_dir", &self.voices_dir).finish_non_exhaustive()
    }
}

impl Kokoro {
    pub fn new(model: KokoroModel, voices_dir: PathBuf, fallback: OovFallback) -> Self {
        Self {
            model,
            voices_dir,
            fallback,
            american: OnceLock::new(),
            british: OnceLock::new(),
            voices: Mutex::new(HashMap::new()),
        }
    }

    /// Speaks `text` in `voice` (see [`parse_voice_spec`]) at `speed`
    /// (0.25 to 4.0). Returns 24 kHz mono samples, edge silence trimmed and
    /// peak-normalised to 0.95, as open-speech returns them.
    pub fn speak(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>, Error> {
        let audio = self.speak_raw(text, voice, speed)?;
        Ok(post::normalize_peak(post::trim_silence(&audio, post::TRIM_THRESHOLD), post::PEAK))
    }

    /// [`Kokoro::speak`] without trimming or normalisation: the model's
    /// output, chunks concatenated.
    pub fn speak_raw(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>, Error> {
        if !(0.25..=4.0).contains(&speed) {
            return Err(Error::Speed(speed));
        }
        let parts = parse_voice_spec(voice)?;
        let dialect = Dialect::for_kokoro_voice(&parts[0].0).ok_or_else(|| {
            Error::Voice(format!("{}: only American (a*) and British (b*) voices are supported", parts[0].0))
        })?;
        let style = self.voice(&parts)?;
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
        Ok(text
            .trim()
            .split('\n')
            .filter(|line| !line.trim().is_empty())
            .flat_map(|line| chunks(g2p.tokens(line)))
            .collect())
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

    fn voice(&self, parts: &[(String, f32)]) -> Result<Arc<Voice>, Error> {
        let key = parts.iter().map(|(id, w)| format!("{id}({w})")).collect::<Vec<_>>().join("+");
        let mut cache = self.voices.lock().map_err(|_| Error::Voice("voice cache lock poisoned".into()))?;
        if let Some(v) = cache.get(&key) {
            return Ok(Arc::clone(v));
        }
        let mut loaded = Vec::with_capacity(parts.len());
        for (id, weight) in parts {
            let voice = match cache.get(id) {
                Some(v) => Arc::clone(v),
                None => {
                    let v = Arc::new(Voice::from_file(&self.voices_dir.join(format!("{id}.bin")))?);
                    cache.insert(id.clone(), Arc::clone(&v));
                    v
                }
            };
            loaded.push((voice, *weight));
        }
        let voice = if loaded.len() == 1 {
            Arc::clone(&loaded[0].0)
        } else {
            let refs: Vec<(&Voice, f32)> = loaded.iter().map(|(v, w)| (v.as_ref(), *w)).collect();
            Arc::new(Voice::blend(&refs)?)
        };
        cache.insert(key, Arc::clone(&voice));
        Ok(voice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_specs_parse_like_open_speech() {
        assert_eq!(parse_voice_spec("alloy").unwrap(), [("af_heart".to_owned(), 1.0)]);
        assert_eq!(
            parse_voice_spec("af_bella(2)+af_sky(1)").unwrap(),
            [("af_bella".to_owned(), 2.0), ("af_sky".to_owned(), 1.0)]
        );
        assert_eq!(parse_voice_spec("af_bella+af_sky").unwrap().len(), 2);
    }

    #[test]
    fn voice_ids_cannot_escape_the_voices_directory() {
        for bad in ["../etc/passwd", "af/heart", "", "af_heart(x)", "af_heart(-1)", "a b"] {
            assert!(parse_voice_spec(bad).is_err(), "{bad:?} should be rejected");
        }
    }
}
