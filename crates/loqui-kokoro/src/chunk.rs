//! Splitting phonemized text into Kokoro-sized pieces.
//!
//! Kokoro takes at most 510 phonemes per call. This ports `KPipeline`'s
//! `en_tokenize` and `waterfall_last` from kokoro 0.9.4: tokens accumulate
//! until the next one would overflow, and the cut then falls after the last
//! sentence end, failing that the last colon or semicolon, failing that the
//! last comma or dash. The chunk boundaries decide where Kokoro breathes,
//! so they follow upstream exactly.

use loqui_g2p::PhonemeToken;

use crate::voice::MAX_PHONEMES;

/// `KPipeline.tokens_to_ps`: each token's phonemes, a space after those
/// followed by whitespace, trimmed.
pub fn tokens_to_ps(tokens: &[PhonemeToken]) -> String {
    let joined: String = tokens.iter().map(|t| t.phonemes.clone() + if t.whitespace { " " } else { "" }).collect();
    joined.trim().to_owned()
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// `KPipeline.waterfall_last`: where to cut `tokens` so the remainder plus
/// the incoming token stays within bounds.
fn waterfall_last(tokens: &[PhonemeToken], next_count: usize) -> usize {
    const WATERFALL: [&str; 3] = ["!.?…", ":;", ",—"];
    const BUMPS: [&str; 2] = [")", "”"];
    for marks in WATERFALL {
        let hit = tokens.iter().rposition(|t| char_len(&t.phonemes) == 1 && marks.contains(t.phonemes.as_str()));
        let Some(mut z) = hit.map(|i| i + 1) else { continue };
        if z < tokens.len() && BUMPS.contains(&tokens[z].phonemes.as_str()) {
            z += 1;
        }
        if next_count.saturating_sub(char_len(&tokens_to_ps(&tokens[..z]))) <= MAX_PHONEMES {
            return z;
        }
    }
    tokens.len()
}

/// `KPipeline.en_tokenize`: the phoneme string of each chunk, in order.
/// Empty chunks are dropped, as `generate_from_tokens` skips them.
pub fn chunks(tokens: Vec<PhonemeToken>) -> Vec<String> {
    let mut out = Vec::new();
    let mut tks: Vec<PhonemeToken> = Vec::new();
    let mut pcount = 0;
    for t in tokens {
        let mut next_ps = t.phonemes.clone() + if t.whitespace { " " } else { "" };
        let next_pcount = pcount + char_len(next_ps.trim_end());
        if next_pcount > MAX_PHONEMES {
            let z = waterfall_last(&tks, next_pcount);
            out.push(tokens_to_ps(&tks[..z]));
            tks.drain(..z);
            pcount = char_len(&tokens_to_ps(&tks));
            if tks.is_empty() {
                next_ps = next_ps.trim_start().to_owned();
            }
        }
        pcount += char_len(&next_ps);
        tks.push(t);
    }
    if !tks.is_empty() {
        out.push(tokens_to_ps(&tks));
    }
    out.retain(|ps| !ps.is_empty());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(phonemes: &str, whitespace: bool) -> PhonemeToken {
        PhonemeToken { text: phonemes.to_owned(), phonemes: phonemes.to_owned(), whitespace }
    }

    #[test]
    fn short_text_is_one_chunk() {
        let chunks = chunks(vec![tok("həlˈO", true), tok("wˈɜɹld", false), tok(".", false)]);
        assert_eq!(chunks, ["həlˈO wˈɜɹld."]);
    }

    #[test]
    fn long_text_is_cut_after_the_last_sentence_end() {
        // Two ~300-phoneme sentences cannot share a 510-phoneme chunk.
        let word = "a".repeat(99);
        let mut tokens = Vec::new();
        for _ in 0..2 {
            tokens.extend([tok(&word, true), tok(&word, true), tok(&word, false), tok(".", true)]);
        }
        let chunks = chunks(tokens);
        assert_eq!(chunks.len(), 2);
        assert!(chunks.iter().all(|c| c.chars().count() <= MAX_PHONEMES && c.ends_with('.')));
    }

    #[test]
    fn silent_tokens_produce_no_chunk() {
        assert!(chunks(vec![tok("", true), tok("", false)]).is_empty());
    }
}
