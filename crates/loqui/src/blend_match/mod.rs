//! Matching a recording with a blend of the stock voices.
//!
//! [`Engine::match_voice`] finds the blend of Kokoro's voices that sounds most
//! like a recording, by speaker-embedding similarity: rank the voices against
//! it, start from the mix of the nearest few that best explains it, then let
//! Kokoro speak candidate blends and keep the moves that bring them closer.
//! See `tools/parity/README.md` ("Speaker similarity") for the measurements
//! the defaults come from.
//!
//! It finds the closest blend available, not a copy: a blend of the stock
//! voices cannot reach every voice, and [`BlendMatch`] says how close it came
//! beside how close the best single voice was.

mod anchors;
mod search;
mod spec;

use std::ops::ControlFlow;

use loqui_audio::Pcm;

use self::search::{Evaluated, Evaluator, refine};
use self::spec::Lead;
use crate::speaker::{self, SpeakerEmbedding};
use crate::{Engine, Error};

/// The most voices a match may mix.
pub const MAX_VOICES: usize = 6;
/// The most blends a match may speak.
pub const MAX_EVALUATIONS: usize = 200;
/// The least speech a recording to match must hold.
pub const MIN_MATCH_SECS: f32 = 3.0;
/// The least, and the most, of a transcript used as the probe.
const PROBE_CHARS: std::ops::RangeInclusive<usize> = 20..=200;

/// The accent a match speaks with: Kokoro's American or British G2P, chosen
/// by the first voice in the blend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Accent {
    /// Whichever of the two sounds closer: one extra blend is spoken to find
    /// out.
    #[default]
    Auto,
    American,
    British,
}

impl Accent {
    fn family(self) -> Option<char> {
        match self {
            Self::Auto => None,
            Self::American => Some('a'),
            Self::British => Some('b'),
        }
    }
}

/// A recording to match, and how hard to try.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MatchRequest {
    /// The recording, in any container loqui-audio decodes. At least
    /// [`MIN_MATCH_SECS`] of speech; the first 30 s are used.
    pub audio: Vec<u8>,
    /// What the recording says, if known. Speaking the same words makes the
    /// comparison sharper, and sets the blend's speed from how fast the
    /// recording speaks; 20 characters or more, and the first 200 or so are
    /// used.
    pub transcript: Option<String>,
    /// How many of the nearest voices to mix, 1 to [`MAX_VOICES`]; 4 by
    /// default.
    pub max_voices: usize,
    /// How many blends to speak, 1 to [`MAX_EVALUATIONS`]; 36 by default.
    /// Each costs one synthesis of a few seconds of speech.
    pub evaluations: usize,
    pub accent: Accent,
}

impl MatchRequest {
    pub fn new(audio: Vec<u8>) -> Self {
        Self { audio, transcript: None, max_voices: 4, evaluations: 36, accent: Accent::Auto }
    }
}

/// The blend found.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BlendMatch {
    /// A voice spec [`crate::SpeakRequest::voice`] takes, in whole percents
    /// with the accent first: `am_michael(65)+am_onyx(35)`.
    pub spec: String,
    /// How alike it and the recording sound: speaker-embedding cosine, 1 for
    /// the same voice.
    pub similarity: f32,
    /// The single stock voice nearest the recording, measured the same way.
    pub closest_voice: String,
    pub closest_similarity: f32,
    /// Every stock voice by its similarity to the recording, nearest first
    /// (against each voice's reference recording, so on a different text).
    pub ranking: Vec<(String, f32)>,
    /// A speaking rate matching the recording's, when a transcript was given.
    pub speed: Option<f32>,
    /// How many blends were spoken.
    pub evaluations: usize,
}

/// How a match is going, for a progress display; returning `Break` from the
/// callback cancels it with [`Error::Cancelled`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MatchProgress {
    /// The stock voices' reference embeddings are being computed: once per
    /// cache, one synthesis per voice.
    Preparing { done: usize, total: usize },
    /// A candidate blend was spoken and measured.
    Evaluated { evaluations: usize, budget: usize, best_similarity: f32 },
}

impl Engine {
    /// The blend of the stock voices that sounds most like `request.audio`.
    /// Needs text-to-speech and [`crate::EngineBuilder::speaker`].
    ///
    /// It speaks: about `request.evaluations` short syntheses, plus 28 the
    /// first time a cache is used, so it takes tens of seconds on a CPU.
    /// Kokoro serves other requests between them. `progress` is told after
    /// each, and cancels the match by returning `Break`.
    pub fn match_voice(&self, request: &MatchRequest, progress: &mut dyn FnMut(&MatchProgress) -> ControlFlow<()>) -> Result<BlendMatch, Error> {
        if !(1..=MAX_VOICES).contains(&request.max_voices) {
            return Err(Error::Invalid(format!("max_voices must be 1 to {MAX_VOICES}")));
        }
        if !(1..=MAX_EVALUATIONS).contains(&request.evaluations) {
            return Err(Error::Invalid(format!("evaluations must be 1 to {MAX_EVALUATIONS}")));
        }
        let tts = self.inner.tts.as_ref().ok_or(Error::Disabled("text-to-speech"))?;
        let speaker_slot = self.inner.speaker.as_ref().ok_or(Error::Disabled("speaker embeddings"))?;

        let target_pcm = loqui_audio::decode_mono_16k(&request.audio, Some(self.inner.max_audio_secs))?;
        let encoder = speaker_slot.get()?;
        let target = speaker::embed(&encoder, &target_pcm.samples, MIN_MATCH_SECS)?;
        let target_secs = speaker::voiced(&target_pcm.samples).0.len() as f32 / loqui_speaker::SAMPLE_RATE as f32;
        // Every pack the search may reach, before any is needed: under
        // `Downloads::Deny` a missing one is reported now, not halfway through.
        self.fetch_voices()?;
        let kokoro = tts.get()?;

        // Speak `text` in `spec` and embed it.
        let embed_spoken = |text: &str, spec: &str, speed: f32| -> Result<(SpeakerEmbedding, f32), Error> {
            let blend = self.resolve(spec)?;
            let samples = kokoro.speak(text, &blend, speed)?;
            let secs = samples.len() as f32 / loqui_kokoro::SAMPLE_RATE as f32;
            let pcm = loqui_audio::resample(Pcm { samples, rate: loqui_kokoro::SAMPLE_RATE }, loqui_speaker::SAMPLE_RATE)?;
            Ok((speaker::embed(&encoder, &pcm.samples, 0.5)?, secs))
        };

        let voices = loqui_kokoro::ENGLISH_VOICES;
        let anchors = self.anchors(&encoder, &mut |voice| embed_spoken(anchors::PROBE, voice, 1.0).map(|(e, _)| e), progress)?;
        let mut ranking: Vec<(&str, f32)> =
            voices.iter().zip(&anchors).map(|(voice, anchor)| (*voice, loqui_speaker::cosine(target.as_slice(), anchor))).collect();
        ranking.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(b.0)));
        let top: Vec<&str> = ranking.iter().take(request.max_voices).map(|(voice, _)| *voice).collect();
        let nearest_of = |family: char| ranking.iter().map(|(voice, _)| *voice).find(|voice| voice.starts_with(family));

        // The probe: the recording's own words when they are known, which
        // sharpens the comparison, and the rate they were spoken at.
        let transcript = request.transcript.as_deref().map(probe_text).filter(|t| PROBE_CHARS.contains(&t.chars().count()));
        let (probe, speed) = match transcript {
            Some(text) => {
                let (_, secs) = embed_spoken(&text, top[0], 1.0)?;
                let speed = (secs / target_secs.max(0.1)).clamp(0.75, 1.35);
                (text, Some((speed * 20.0).round() / 20.0))
            }
            None => (anchors::PROBE.to_owned(), None),
        };

        let start = {
            let index = |voice: &str| voices.iter().position(|v| *v == voice).expect("ranked voices are stock voices");
            let refs: Vec<&[f32]> = top.iter().map(|voice| anchors[index(voice)].as_slice()).collect();
            spec::lattice(&search::least_squares(&refs, target.as_slice()), 5)
        };
        let forced = request.accent.family();
        let lead = |percents: &[u32]| -> Lead<'static> {
            let heaviest = top[percents.iter().enumerate().max_by_key(|(i, p)| (**p, std::cmp::Reverse(*i))).map_or(0, |(i, _)| i)];
            match forced {
                Some(family) if !heaviest.starts_with(family) => nearest_of(family).map_or(Lead::Heaviest, Lead::Voice),
                _ => Lead::Heaviest,
            }
        };

        // Leave one synthesis for trying the other accent.
        let reserve = usize::from(forced.is_none() && request.evaluations > 1);
        let mut objective = |spec: &str| embed_spoken(&probe, spec, speed.unwrap_or(1.0)).map(|(e, _)| e.similarity(&target));
        let mut report = |e: Evaluated| progress(&MatchProgress::Evaluated { evaluations: e.evaluations, budget: e.budget, best_similarity: e.best });
        let mut eval = Evaluator::new(request.evaluations - reserve, &mut objective, &mut report);

        let single: Vec<u32> = (0..top.len()).map(|i| if i == 0 { 100 } else { 0 }).collect();
        let closest_spec = spec::format(&top, &single, lead(&single));
        let closest_similarity = eval.score(&closest_spec)?.unwrap_or(f32::NEG_INFINITY);
        let mut best = refine(&top, start, lead, &mut eval)?;
        if closest_similarity > best.similarity {
            best = search::Found { percents: single, spec: closest_spec, similarity: closest_similarity };
        }
        if forced.is_none() {
            eval.raise_budget(reserve);
            let heaviest = best.spec.split(['(', '+']).next().unwrap_or_default();
            let other = if heaviest.starts_with('b') { 'a' } else { 'b' };
            if let Some(voice) = nearest_of(other) {
                let spec = spec::format(&top, &best.percents, Lead::Voice(voice));
                if let Some(similarity) = eval.score(&spec)?
                    && similarity > best.similarity + 0.001
                {
                    best = search::Found { percents: best.percents, spec, similarity };
                }
            }
        }

        Ok(BlendMatch {
            spec: best.spec,
            similarity: best.similarity,
            closest_voice: top[0].to_owned(),
            closest_similarity,
            ranking: ranking.iter().map(|(voice, similarity)| ((*voice).to_owned(), *similarity)).collect(),
            speed,
            evaluations: eval.evaluations(),
        })
    }

    /// The stock voices' reference embeddings, from the cache or, the first
    /// time, by speaking [`anchors::PROBE`] in each.
    fn anchors(
        &self,
        encoder: &loqui_speaker::SpeakerEncoder,
        embed: &mut dyn FnMut(&str) -> Result<SpeakerEmbedding, Error>,
        progress: &mut dyn FnMut(&MatchProgress) -> ControlFlow<()>,
    ) -> Result<Vec<Vec<f32>>, Error> {
        let voices = loqui_kokoro::ENGLISH_VOICES;
        let kokoro_sha256 = self.inner.tts_variant.map_or("", |variant| variant.file().sha256);
        let key = anchors::key(kokoro_sha256, crate::models::SPEAKER_FILE.sha256, voices);
        let path = anchors::path(&self.inner.cache, &key);
        if let Some(cached) = anchors::read(&path, voices.len(), encoder.dim()) {
            return Ok(cached);
        }
        let mut out = Vec::with_capacity(voices.len());
        for (done, voice) in voices.iter().enumerate() {
            if progress(&MatchProgress::Preparing { done, total: voices.len() }).is_break() {
                return Err(Error::Cancelled);
            }
            out.push(embed(voice)?.as_slice().to_vec());
        }
        // A cache that cannot be written (a read-only volume) costs the next
        // match the same work, and nothing else.
        if let Err(e) = anchors::write(&path, &out) {
            tracing::warn!("voice anchors not cached: {e}");
        }
        Ok(out)
    }
}

/// A transcript cut to a probe: the sentences that fit in the first 200
/// characters, or the first 200 characters when no sentence ends there.
fn probe_text(transcript: &str) -> String {
    let text = transcript.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.chars().count() <= *PROBE_CHARS.end() {
        return text;
    }
    let cut: String = text.chars().take(*PROBE_CHARS.end()).collect();
    match cut.rfind(['.', '!', '?']) {
        Some(end) if end + 1 >= *PROBE_CHARS.start() => cut[..=end].to_owned(),
        _ => cut,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Downloads, SpeakerConfig};

    fn quiet(_: &MatchProgress) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }

    /// A bad request is refused before anything is decoded, loaded or
    /// fetched: these engines have no weights and may not download any.
    #[test]
    fn a_bad_request_is_refused_before_anything_loads() {
        let cache = std::env::temp_dir().join(format!("loqui-match-refuse-{}", std::process::id()));
        let engine = Engine::builder().cache_dir(&cache).downloads(Downloads::Deny).speaker(Some(SpeakerConfig::default())).build().unwrap();
        for (max_voices, evaluations, want) in [(0, 36, "max_voices"), (7, 36, "max_voices"), (4, 0, "evaluations"), (4, 201, "evaluations")] {
            let mut request = MatchRequest::new(Vec::new());
            request.max_voices = max_voices;
            request.evaluations = evaluations;
            let err = engine.match_voice(&request, &mut quiet).unwrap_err();
            assert!(matches!(&err, Error::Invalid(m) if m.contains(want)), "{err:?}");
        }
        let without_speaker = Engine::builder().cache_dir(&cache).downloads(Downloads::Deny).build().unwrap();
        let err = without_speaker.match_voice(&MatchRequest::new(Vec::new()), &mut quiet).unwrap_err();
        assert!(matches!(err, Error::Disabled("speaker embeddings")), "{err:?}");
        assert!(!cache.exists(), "nothing was fetched or created");
    }

    #[test]
    fn a_long_transcript_is_cut_at_a_sentence() {
        let short = "Hello there, this is a short note.";
        assert_eq!(probe_text(short), short);
        let long = format!("{} And then it went on. {}", "Word ".repeat(30).trim(), "More words ".repeat(20));
        let cut = probe_text(&long);
        assert!(cut.ends_with("went on."), "{cut}");
        assert!(cut.chars().count() <= 200);
        let run_on = "word ".repeat(80);
        assert_eq!(probe_text(&run_on).chars().count(), 200);
    }
}
