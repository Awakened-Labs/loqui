//! Phonemizes stdin line by line, for parity checks against Python misaki.
//!
//!     phonemize [--gb] [--word | --spell-out | --tokens | --tags] < lines.txt
//!
//! `--word` runs only the neural fallback model on each line, which is how
//! it is compared with the PyTorch checkpoint. `--tokens` prints each line's
//! tokens as a JSON array of `[text, space_after]`, for comparison with
//! spaCy; `--tags` adds each token's part-of-speech tag. Output is one line
//! per input line, in the same order.

use std::io::{BufRead, Write};

use loqui_g2p::tagger::Tagger;
use loqui_g2p::tokenize::Tokenizer;
use loqui_g2p::{Dialect, G2p, NeuralG2p, OovFallback};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    let dialect = if has("--gb") { Dialect::British } else { Dialect::American };
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout());

    if has("--tags") {
        let (tokenizer, tagger) = (Tokenizer::english()?, Tagger::english()?);
        for line in stdin.lock().lines() {
            let tokens = tokenizer.tokenize(&line?);
            let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
            let tagged: Vec<(&str, String)> = words.iter().copied().zip(tagger.tag(&words)).collect();
            writeln!(out, "{}", serde_json::to_string(&tagged)?)?;
        }
    } else if has("--tokens") {
        let tokenizer = Tokenizer::english()?;
        for line in stdin.lock().lines() {
            let tokens: Vec<(String, bool)> =
                tokenizer.tokenize(&line?).into_iter().map(|t| (t.text, t.space_after)).collect();
            writeln!(out, "{}", serde_json::to_string(&tokens)?)?;
        }
    } else if has("--word") {
        let model = NeuralG2p::load(dialect)?;
        for line in stdin.lock().lines() {
            writeln!(out, "{}", model.phonemize(line?.trim()).unwrap_or_default())?;
        }
    } else {
        let fallback = if has("--spell-out") { OovFallback::SpellOut } else { OovFallback::Neural };
        let g2p = G2p::new(dialect, fallback)?;
        for line in stdin.lock().lines() {
            writeln!(out, "{}", g2p.phonemize(&line?).trim_end())?;
        }
    }
    Ok(())
}
