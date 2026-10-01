# loqui-g2p

English grapheme-to-phoneme conversion for Kokoro, GPL-free: a Rust port of
misaki's English pipeline (spaCy tokenization, Penn Treebank tags, the misaki
gold and silver lexicons, heteronyms, stress, numbers), with a small embedded
neural model for unknown words where upstream calls eSpeak NG.

```rust
use loqui_g2p::{Dialect, G2p, OovFallback};

let g2p = G2p::new(Dialect::American, OovFallback::Neural)?;
assert_eq!(g2p.phonemize("Hello world."), "həlˈO wˈɜɹld.");
```

The lexicons and G2P weights are embedded, so there is nothing to download.

Part of [loqui](https://github.com/Awakened-Labs/loqui): local Kokoro
text-to-speech and Whisper speech-to-text for Rust. Most programs want the
[`loqui`](https://crates.io/crates/loqui) facade rather than this crate
directly. MIT; see `NOTICE` for third-party attributions.
