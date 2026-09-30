//! English grapheme-to-phoneme conversion for Kokoro, GPL-free.
//!
//! Kokoro's voices were trained on phonemes from misaki, hexgrad's G2P, so
//! this crate reproduces misaki's English pipeline: spaCy's tokenization,
//! Penn Treebank tags, the misaki gold and silver lexicons, heteronym
//! selection, stress rules, morphology, and number, currency and year
//! reading. Each stage is checked against its Python original by the parity
//! harness in `tools/parity`.
//!
//! Where upstream hands unknown words to eSpeak NG (GPL-3.0), this crate
//! uses a small embedded neural model trained on the same lexicons
//! ([`OovFallback::Neural`]). It links no GPL code.
//!
//! ```
//! use loqui_g2p::{Dialect, G2p, OovFallback};
//!
//! let g2p = G2p::new(Dialect::American, OovFallback::Neural)?;
//! assert_eq!(g2p.phonemize("Hello world."), "həlˈO wˈɜɹld.");
//! # Ok::<(), loqui_g2p::Error>(())
//! ```

mod en;
mod lexicon;
mod neural;
pub mod numbers;
pub mod tagger;
pub mod tokenize;

pub use neural::{MAX_WORD_CHARS, NeuralG2p};

use en::{Lexicon, MToken};
use tagger::Tagger;
use tokenize::Tokenizer;

/// What misaki writes for a word it could not phonemize at all.
pub const UNKNOWN: &str = "❓";

/// One spoken word and its phonemes, as Kokoro's chunker consumes them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhonemeToken {
    /// The source text of the word.
    pub text: String,
    /// Its phonemes; empty for silent tokens (currency signs, `#`) and for
    /// anything nothing could phonemize.
    pub phonemes: String,
    /// Whether whitespace followed the word in the source.
    pub whitespace: bool,
}

/// Which English the lexicons, rules and fallback model follow.
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
    /// `j`, `p` and `z` voices need other G2P engines).
    pub fn for_kokoro_voice(voice: &str) -> Option<Self> {
        match voice.chars().next()? {
            'a' => Some(Self::American),
            'b' => Some(Self::British),
            _ => None,
        }
    }
}

/// What happens to a word that no lexicon entry, morphology rule or number
/// rule covers: a name, a coinage, a typo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OovFallback {
    /// The embedded character-level BART model (Apache-2.0), trained on the
    /// misaki lexicons. It pronounces unseen words as words, and scores
    /// slightly better than eSpeak NG against CMUdict on held-out names.
    #[default]
    Neural,
    /// Spell the word letter by letter. Predictable, and right for
    /// acronyms, but it reads "zorbulate" as nine letter names.
    SpellOut,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Embedded data (model weights, lexicons, tokenizer rules) could not
    /// be read. This is a build defect, never a runtime condition.
    #[error("embedded G2P data is malformed: {0}")]
    Weights(String),
}

/// A ready-to-use phonemizer. Construction parses the lexicons (tens of
/// megabytes of JSON), so build one and share it: it is `Send + Sync`.
pub struct G2p {
    dialect: Dialect,
    fallback: OovFallback,
    tokenizer: Tokenizer,
    tagger: Tagger,
    lexicon: Lexicon,
    neural: Option<NeuralG2p>,
}

impl std::fmt::Debug for G2p {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("G2p").field("dialect", &self.dialect).field("fallback", &self.fallback).finish_non_exhaustive()
    }
}

impl G2p {
    pub fn new(dialect: Dialect, fallback: OovFallback) -> Result<Self, Error> {
        Ok(Self {
            dialect,
            fallback,
            tokenizer: Tokenizer::english()?,
            tagger: Tagger::english()?,
            lexicon: lexicon::load(dialect)?,
            neural: match fallback {
                OovFallback::Neural => Some(NeuralG2p::load(dialect)?),
                OovFallback::SpellOut => None,
            },
        })
    }

    pub fn dialect(&self) -> Dialect {
        self.dialect
    }

    /// Converts `text` to Kokoro phonemes. Punctuation is kept, because
    /// Kokoro uses it for prosody. misaki's inline markup is honoured:
    /// `[word](/phonemes/)` overrides a pronunciation, `[word](+2)` sets
    /// stress, and `[5](#a#)` passes number-reading flags.
    pub fn phonemize(&self, text: &str) -> String {
        self.run(text, UNKNOWN).into_iter().map(|tk| tk.phonemes.unwrap_or_else(|| UNKNOWN.to_owned()) + &tk.whitespace).collect()
    }

    /// The same conversion, one token per spoken word, with nothing marked
    /// unknown: a word that cannot be phonemized is simply silent. This is
    /// what Kokoro's own pipeline feeds its chunker (misaki with `unk=''`).
    pub fn tokens(&self, text: &str) -> Vec<PhonemeToken> {
        self.run(text, "")
            .into_iter()
            .map(|tk| PhonemeToken { phonemes: tk.phonemes.unwrap_or_default(), whitespace: !tk.whitespace.is_empty(), text: tk.text })
            .collect()
    }

    fn run(&self, text: &str, unk: &str) -> Vec<MToken> {
        let word_fallback = |word: &str| -> Option<String> {
            // A short word with no vowel letter ("kg", "mph", "SQL" in
            // lowercase) is an abbreviation; eSpeak spells those out too.
            let vowelless = !word.chars().any(|c| "aeiouyAEIOUY".contains(c));
            if vowelless
                && word.chars().count() <= 5
                && let Some(spelled) = self.lexicon.spell(word)
            {
                return Some(spelled);
            }
            match &self.neural {
                // A word the model cannot encode at all ("日本") is dropped
                // rather than read as an unknown marker.
                Some(model) => Some(model.phonemize(word).unwrap_or_default()),
                None => self.lexicon.spell(word),
            }
        };
        let fallback = |word: &str| self.lexicon.compose_fallback(word, &word_fallback);
        en::g2p(&self.lexicon, text, |t| self.tag(t), &fallback, unk)
    }

    fn tag(&self, text: &str) -> Vec<MToken> {
        let tokens = self.tokenizer.tokenize(text);
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        let tags = self.tagger.tag(&words);
        tokens
            .into_iter()
            .zip(tags)
            // spaCy tags URLs and e-mail addresses ADD, and misaki then reads
            // their "." as "dot" and "/" as "slash". The pattern decides
            // that more reliably than the statistical tagger does.
            .map(|(t, tag)| {
                let tag = if self.tokenizer.is_url(&t.text) {
                    "ADD".to_owned()
                } else if is_numeral(&t.text) {
                    // A numeral is CD whatever its neighbours say, and misaki
                    // reads money and plain numbers only from CD tokens.
                    "CD".to_owned()
                } else {
                    tag
                };
                (t, tag)
            })
            .map(|(t, tag)| MToken {
                whitespace: if t.space_after { " ".to_owned() } else { String::new() },
                text: t.text,
                tag,
                is_head: true,
                ..MToken::default()
            })
            .collect()
    }
}

/// Digits with optional thousands separators, a decimal point and a leading
/// minus: "12", "1,234,567.89", "-5", "4.7".
fn is_numeral(text: &str) -> bool {
    let body = text.strip_prefix('-').unwrap_or(text);
    body.starts_with(|c: char| c.is_ascii_digit()) && body.chars().all(|c| c.is_ascii_digit() || c == ',' || c == '.')
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
        assert_eq!(neural.phonemize("zorbulate"), "zˈɔɹbjəlˌAt");
        assert_ne!(neural.phonemize("zorbulate"), spelled.phonemize("zorbulate"));
    }

    #[test]
    fn tokens_carry_phonemes_and_spacing_per_word() {
        let g2p = G2p::new(Dialect::American, OovFallback::Neural).unwrap();
        let tokens = g2p.tokens("It costs $5.");
        let shown: Vec<(&str, &str, bool)> = tokens.iter().map(|t| (t.text.as_str(), t.phonemes.as_str(), t.whitespace)).collect();
        assert_eq!(
            shown,
            [("It", "ˌɪt", true), ("costs", "kˈɔsts", true), ("$", "", false), ("5", "fˈIv dˈɑləɹz", false), (".", ".", false)]
        );
    }

    #[test]
    fn inline_markup_overrides_pronunciation() {
        let g2p = G2p::new(Dialect::American, OovFallback::Neural).unwrap();
        assert_eq!(g2p.phonemize("[Kokoro](/kˈOkəɹO/) speaks."), "kˈOkəɹO spˈiks.");
    }
}
