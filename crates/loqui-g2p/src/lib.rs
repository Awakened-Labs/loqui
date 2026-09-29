//! English grapheme-to-phoneme conversion for Kokoro, GPL-free by default.
//!
//! Text goes through [misaki-rs], a port of Kokoro's own G2P: the misaki
//! gold and silver lexicons, an averaged-perceptron POS tagger for
//! heteronyms ("read", "lead", "object"), suffix morphology and number
//! expansion. The output is Kokoro's phoneme alphabet.
//!
//! What differs from upstream misaki is the fallback for words no lexicon
//! or rule covers. Upstream hands those to eSpeak NG, which is GPL-3.0 and
//! cannot be linked into permissively licensed or proprietary programs.
//! Here the default is a small embedded neural model trained on the same
//! lexicons (see [`OovFallback::Neural`]), and eSpeak NG is an opt-in
//! feature.
//!
//! [misaki-rs]: https://crates.io/crates/misaki-rs
//!
//! ```
//! use loqui_g2p::{Dialect, G2p, OovFallback};
//!
//! let g2p = G2p::new(Dialect::American, OovFallback::Neural)?;
//! assert!(g2p.phonemize("Hello world.")?.starts_with("həlˈO wˈɜɹld"));
//! # Ok::<(), loqui_g2p::Error>(())
//! ```

mod lexicon;
mod neural;

use std::sync::Arc;

use misaki_rs::fallback::FallbackError;

pub use neural::{MAX_WORD_CHARS, NeuralG2p};

/// Which English the lexicons, tagger rules and fallback model follow.
///
/// Kokoro voices encode their dialect in the first letter of the voice id:
/// `a` (`af_heart`) is American, `b` (`bf_emma`) is British.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dialect {
    American,
    British,
}

impl Dialect {
    /// The dialect a Kokoro voice id implies, or `None` for a voice whose
    /// language this crate does not phonemize (Kokoro's `e`, `f`, `h`, `i`,
    /// `j`, `p` and `z` voices need eSpeak NG or other G2P engines).
    pub fn for_kokoro_voice(voice: &str) -> Option<Self> {
        match voice.chars().next()? {
            'a' => Some(Self::American),
            'b' => Some(Self::British),
            _ => None,
        }
    }

    fn language(self) -> misaki_rs::Language {
        match self {
            Self::American => misaki_rs::Language::EnglishUS,
            Self::British => misaki_rs::Language::EnglishGB,
        }
    }
}

/// What happens to a word that no lexicon entry, morphology rule or number
/// rule covers: a name, a coinage, a typo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OovFallback {
    /// The embedded character-level BART model (Apache-2.0), trained on the
    /// misaki lexicons. The default: it pronounces unseen words as words
    /// and links no GPL code.
    #[default]
    Neural,
    /// Spell the word letter by letter. Predictable, and right for
    /// acronyms, but it reads "zorbulate" as nine letter names.
    SpellOut,
    /// eSpeak NG, linked in. GPL-3.0: a binary built with the `espeak`
    /// feature is subject to the GPL. Matches upstream misaki's behaviour.
    #[cfg(feature = "espeak")]
    Espeak,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The embedded G2P checkpoint could not be read. This is a build
    /// defect, never a runtime condition.
    #[error("embedded G2P weights are malformed: {0}")]
    Weights(String),
    #[error("phonemization failed: {0}")]
    Phonemize(String),
}

/// A ready-to-use phonemizer. Construction parses the lexicons (tens of
/// megabytes of JSON), so build one and share it: it is `Send + Sync`.
pub struct G2p {
    inner: misaki_rs::G2P,
    dialect: Dialect,
    fallback: OovFallback,
}

impl std::fmt::Debug for G2p {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("G2p").field("dialect", &self.dialect).field("fallback", &self.fallback).finish_non_exhaustive()
    }
}

impl G2p {
    pub fn new(dialect: Dialect, fallback: OovFallback) -> Result<Self, Error> {
        let language = dialect.language();
        let inner = match fallback {
            OovFallback::Neural => {
                let model = Arc::new(NeuralG2p::load(dialect)?);
                misaki_rs::G2P::with_fallback(language, Some(Box::new(NeuralFallback(model))))
            }
            OovFallback::SpellOut => misaki_rs::G2P::with_fallback(language, None),
            // misaki-rs installs its own eSpeak fallback when the feature is on.
            #[cfg(feature = "espeak")]
            OovFallback::Espeak => misaki_rs::G2P::new(language),
        };
        let mut inner = inner;
        let (golds, silvers) = lexicon::load(dialect)?;
        inner.lexicon.golds = golds;
        inner.lexicon.silvers = silvers;
        Ok(Self { inner, dialect, fallback })
    }

    pub fn dialect(&self) -> Dialect {
        self.dialect
    }

    /// Converts `text` to Kokoro phonemes. Punctuation is kept, because
    /// Kokoro uses it for prosody.
    pub fn phonemize(&self, text: &str) -> Result<String, Error> {
        self.inner.g2p(text).map(|(phonemes, _)| phonemes).map_err(|e| Error::Phonemize(e.to_string()))
    }
}

/// Adapts [`NeuralG2p`] to misaki-rs's fallback hook.
struct NeuralFallback(Arc<NeuralG2p>);

impl misaki_rs::Fallback for NeuralFallback {
    fn phonemize(&self, word: &str) -> Result<String, FallbackError> {
        // misaki-rs aborts the whole utterance when a fallback errors. A word
        // the model cannot encode at all ("日本") is better dropped than
        // allowed to silence the sentence around it.
        Ok(self.0.phonemize(word).unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn g2p_is_shareable_across_threads() {
        assert_send_sync::<G2p>();
    }

    #[test]
    fn dialect_follows_the_kokoro_voice_prefix() {
        assert_eq!(Dialect::for_kokoro_voice("af_heart"), Some(Dialect::American));
        assert_eq!(Dialect::for_kokoro_voice("bm_george"), Some(Dialect::British));
        assert_eq!(Dialect::for_kokoro_voice("jf_alpha"), None);
        assert_eq!(Dialect::for_kokoro_voice(""), None);
    }

    #[test]
    fn neural_fallback_pronounces_unknown_words_as_words() {
        let neural = G2p::new(Dialect::American, OovFallback::Neural).unwrap();
        let spelled = G2p::new(Dialect::American, OovFallback::SpellOut).unwrap();
        let word = "zorbulate";
        let n = neural.phonemize(word).unwrap();
        let n = n.trim();
        let s = spelled.phonemize(word).unwrap();
        assert!(!n.contains(' '), "neural output should be one word, got {n:?}");
        assert!(s.split_whitespace().count() >= word.len(), "spell-out should be letter names, got {s:?}");
    }
}
